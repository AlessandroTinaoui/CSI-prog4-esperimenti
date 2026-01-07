import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import flwr as fl
from flwr.client import NumPyClient
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def pseudo_huber_grad_hess(residual: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    loss = delta^2 (sqrt(1 + (r/delta)^2) - 1)
    grad = r / sqrt(1 + (r/d)^2)
    hess = 1 / (1 + (r/d)^2)^(3/2)
    """
    d = float(delta)
    z = residual / d
    s = np.sqrt(1.0 + z * z)
    grad = residual / s
    hess = 1.0 / (s ** 3)
    return grad, hess


def parse_model_bytes(parameters: List[np.ndarray]) -> Dict:
    if not parameters or parameters[0].size == 0:
        return {"base_score": 0.0, "stumps": [], "selected_features": None, "bin_edges": None}
    blob = np.array(parameters[0], dtype=np.uint8).tobytes()
    try:
        return json.loads(blob.decode("utf-8"))
    except Exception:
        return {"base_score": 0.0, "stumps": [], "selected_features": None, "bin_edges": None}


def predict_stumps(X: pd.DataFrame, model: Dict) -> np.ndarray:
    pred = np.full(shape=(len(X),), fill_value=float(model.get("base_score", 0.0)), dtype=float)
    stumps = model.get("stumps", []) or []
    for s in stumps:
        feat = s["feature"]
        thr = float(s["thr"])
        wL = float(s["w_left"])
        wR = float(s["w_right"])
        lr = float(s.get("lr", 1.0))
        xcol = X[feat].to_numpy(dtype=float, copy=False)
        pred += lr * np.where(xcol <= thr, wL, wR)
    return pred


class HistClient(NumPyClient):
    def __init__(self, cid: int, data_path: str):
        self.cid = int(cid)
        self.data = pd.read_csv(data_path, sep=",").dropna()

        self.logger = logging.getLogger(f"client_{self.cid}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(LOGS_DIR / f"client_{self.cid}_log.txt")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        cols_to_drop = ["day", "client_id", "user_id", "source_file"]
        self.cols_to_drop = [c for c in cols_to_drop if c in self.data.columns]

        if "label" not in self.data.columns:
            raise ValueError("Target column 'label' non trovata nel dataset")

        X_full = self.data.drop(columns=self.cols_to_drop + ["label"])
        y_full = self.data["label"].astype(float)

        self.feature_names_full = X_full.columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42
        )

    # ---- round 1: feature selection (come prima, ma senza xgboost: facciamo una importance semplice)
    def _simple_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        # importanza grezza: |corr| con y (fallback robusto e velocissimo)
        # Se vuoi, puoi rimettere XGBoost locale come nel tuo vecchio codice.
        imps = []
        yv = y.to_numpy(dtype=float)
        for c in X.columns:
            xv = X[c].to_numpy(dtype=float)
            if np.std(xv) < 1e-12:
                imps.append(0.0)
                continue
            corr = np.corrcoef(xv, yv)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            imps.append(abs(float(corr)))
        return np.array(imps, dtype=float)

    # ---- round 2: binning locale (quantili)
    def _local_bin_edges(self, X: pd.DataFrame, selected_features: List[str], n_bins: int) -> Dict[str, List[float]]:
        edges: Dict[str, List[float]] = {}
        qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
        for feat in selected_features:
            col = X[feat].to_numpy(dtype=float)
            # quantili robusti; se costante -> tutti uguali
            try:
                qv = np.quantile(col, qs)
            except Exception:
                qv = np.array([np.min(col)] * (len(qs) - 1) + [np.max(col)], dtype=float)
            qv = np.maximum.accumulate(qv)
            edges[feat] = qv.tolist()
        return edges

    # ---- round >=3: istogrammi G/H per stump
    def _build_histograms(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Dict,
        bin_edges: Dict[str, List[float]],
        selected_features: List[str],
        huber_delta: float,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], float, float, float]:
        # predizioni correnti
        pred = predict_stumps(X, model)
        residual = pred - y.to_numpy(dtype=float)

        g, h = pseudo_huber_grad_hess(residual, delta=huber_delta)

        total_G = float(np.sum(g))
        total_H = float(np.sum(h))

        # MAE (metrica) sul train locale
        train_mae = float(mean_absolute_error(y, pred))

        hist_g: Dict[str, List[float]] = {}
        hist_h: Dict[str, List[float]] = {}

        for feat in selected_features:
            edges = bin_edges.get(feat)
            if not edges or len(edges) < 3:
                continue
            B = len(edges) - 1
            xcol = X[feat].to_numpy(dtype=float, copy=False)

            # assegna bin index
            # np.searchsorted ritorna in [0..len(edges)], convertiamo in [0..B-1]
            idx = np.searchsorted(np.array(edges, dtype=float), xcol, side="right") - 1
            idx = np.clip(idx, 0, B - 1)

            g_bins = np.zeros(B, dtype=float)
            h_bins = np.zeros(B, dtype=float)

            # accumulo per bin
            np.add.at(g_bins, idx, g)
            np.add.at(h_bins, idx, h)

            hist_g[feat] = g_bins.tolist()
            hist_h[feat] = h_bins.tolist()

        return hist_g, hist_h, total_G, total_H, train_mae

    def fit(self, parameters, config):
        phase = (config or {}).get("phase", "train")

        if phase == "fs":
            self.logger.info("Round 1: FEATURE SELECTION.")
            imp = self._simple_importance(self.X_train, self.y_train)
            metrics = {
                "local_feature_importance": json.dumps(imp.tolist()),
                "feature_names": json.dumps(self.feature_names_full),
            }
            return [], len(self.X_train), metrics

        selected = (config or {}).get("selected_features", None)
        if selected is None:
            selected_features = list(self.feature_names_full)
        else:
            selected_features = json.loads(selected) if isinstance(selected, str) else list(selected)

        # allinea all'ordine locale
        selected_set = set(selected_features)
        selected_features = [f for f in self.feature_names_full if f in selected_set]

        if phase == "binning":
            n_bins = int((config or {}).get("n_bins", 64))
            edges = self._local_bin_edges(self.X_train, selected_features, n_bins=n_bins)
            metrics = {"bin_edges": json.dumps(edges)}
            return [], len(self.X_train), metrics

        # phase == "train"
        model = parse_model_bytes(parameters)
        bin_edges = (config or {}).get("bin_edges", None)
        if bin_edges is None:
            # se non arrivano, prova dal modello
            bin_edges = model.get("bin_edges", None)
        else:
            bin_edges = json.loads(bin_edges) if isinstance(bin_edges, str) else bin_edges

        if not isinstance(bin_edges, dict):
            # non possiamo fare hist
            metrics = {"train_mae": float("nan")}
            return [], len(self.X_train), metrics

        huber_delta = float((config or {}).get("huber_delta", 1.0))

        Xtr = self.X_train[selected_features]
        ytr = self.y_train

        hist_g, hist_h, total_G, total_H, train_mae = self._build_histograms(
            X=Xtr,
            y=ytr,
            model=model,
            bin_edges=bin_edges,
            selected_features=selected_features,
            huber_delta=huber_delta,
        )

        metrics = {
            "hist_g": json.dumps(hist_g),
            "hist_h": json.dumps(hist_h),
            "total_G": float(total_G),
            "total_H": float(total_H),
            "train_mae": float(train_mae),
        }

        # Non mandiamo Parameters: mandiamo tutto in metrics (server aggrega)
        return [], len(Xtr), metrics

    def evaluate(self, parameters, config):
        model = parse_model_bytes(parameters)

        selected = (config or {}).get("selected_features", None)
        if selected is None:
            selected_features = model.get("selected_features") or self.feature_names_full
        else:
            selected_features = json.loads(selected) if isinstance(selected, str) else list(selected)

        selected_set = set(selected_features)
        selected_features = [f for f in self.feature_names_full if f in selected_set]

        Xte = self.X_test[selected_features]
        yte = self.y_test

        pred = predict_stumps(Xte, model)
        mae = float(mean_absolute_error(yte, pred))

        return mae, len(Xte), {"eval_mae": mae}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_app.py <client_id> [optional_data_path]")
        sys.exit(1)

    client_id = int(sys.argv[1])

    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    else:
        # fallback coerente col tuo layout
        CLIENTS_DATA_DIR = Path(__file__).resolve().parent / "clients_data"
        data_path = str(CLIENTS_DATA_DIR / f"group{client_id}_merged_clean.csv")

    if not os.path.exists(data_path):
        print(f"ERRORE: Data file not found: {data_path}")
        sys.exit(1)

    client = HistClient(cid=client_id, data_path=data_path)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
