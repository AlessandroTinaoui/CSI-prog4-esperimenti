import os
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import flwr as fl
from flwr.client import NumPyClient
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # extratreemodel/
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CLIENTS_DATA_DIR = Path(__file__).resolve().parent / "clients_data"


def _bytes_from_ndarrays(parameters: List[np.ndarray]) -> Optional[bytes]:
    if not parameters:
        return None
    arr = parameters[0]
    if arr is None or getattr(arr, "size", 0) == 0:
        return None
    return np.array(arr, dtype=np.uint8).tobytes()


def _make_extratrees(random_state: int) -> ExtraTreesRegressor:
    # max_features DEVE essere float/int/None/"sqrt"/"log2" (non stringa)
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=1,
        max_features=1.0,
        bootstrap=False,
        min_samples_split=2,
        min_samples_leaf=1,
    )


class ExtraTreesClient(NumPyClient):
    def __init__(self, cid: int, data_path: str):
        self.cid = int(cid)
        self.data_path = data_path
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

        self.target_col = "label"
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' non trovata nel dataset {data_path}")

        self.X_full = self.data.drop(columns=self.cols_to_drop + [self.target_col])
        self.y_full = self.data[self.target_col]
        self.feature_names_full = self.X_full.columns.tolist()

        self.X_train_full, self.X_test_full, self.y_train, self.y_test = train_test_split(
            self.X_full,
            self.y_full,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

        self.selected_features: Optional[List[str]] = None
        self.last_local_model: Optional[ExtraTreesRegressor] = None

        self.logger.info(f"Loaded CSV: {data_path}")
        self.logger.info(f"X_full shape={self.X_full.shape}, y_full shape={self.y_full.shape}")

    def _ensure_columns(self, cols: List[str]) -> None:
        # crea colonne mancanti a 0 (garantisce compatibilità tra client)
        for c in cols:
            if c not in self.X_train_full.columns:
                self.X_train_full[c] = 0
            if c not in self.X_test_full.columns:
                self.X_test_full[c] = 0

    def _train_for_importance(self, X_train_df: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        model = _make_extratrees(random_state=42 + self.cid)
        model.fit(X_train_df, y_train)
        return np.array(model.feature_importances_, dtype=float)

    def fit(self, parameters, config):
        phase = (config or {}).get("phase", "train")

        # -------- Round 1: Feature Selection --------
        if phase == "fs":
            self.logger.info("Round 1: FEATURE SELECTION.")
            importances = self._train_for_importance(self.X_train_full, self.y_train)
            metrics = {
                "local_feature_importance": json.dumps(importances.tolist()),
                "feature_names": json.dumps(self.feature_names_full),
            }
            return [], len(self.X_train_full), metrics

        # -------- Round >=2: Train --------
        selected = (config or {}).get("selected_features", None)
        if selected is None:
            selected_features = list(self.feature_names_full)
        else:
            # ✅ FIX: ordine IDENTICO a quello del server
            selected_features = json.loads(selected) if isinstance(selected, str) else list(selected)

        # ✅ crea colonne mancanti e usa quell’ordine
        self._ensure_columns(selected_features)
        self.selected_features = list(selected_features)

        X_train = self.X_train_full[self.selected_features]

        model = _make_extratrees(random_state=1000 + self.cid)
        model.fit(X_train, self.y_train)
        self.last_local_model = model

        y_pred_train = model.predict(X_train)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)

        model_bytes = pickle.dumps(model)
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)

        metrics = {
            "train_mae": float(mae_train),
            "n_features": int(len(self.selected_features)),
        }
        return [model_array], len(X_train), metrics

    def evaluate(self, parameters, config):
        selected = (config or {}).get("selected_features", None)
        if selected is None:
            selected_features = self.selected_features or list(self.feature_names_full)
        else:
            # ✅ FIX: ordine IDENTICO a quello del server anche in evaluate
            selected_features = json.loads(selected) if isinstance(selected, str) else list(selected)

        self._ensure_columns(selected_features)
        X_test = self.X_test_full[selected_features]

        model_bytes = _bytes_from_ndarrays(parameters)
        if not model_bytes:
            if self.last_local_model is None:
                return float("nan"), len(X_test), {"eval_mae": float("nan")}
            y_pred = self.last_local_model.predict(X_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            return float(mae), len(X_test), {"eval_mae": float(mae), "model_used": "local"}

        try:
            ensemble = pickle.loads(model_bytes)
            y_pred = ensemble.predict(X_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            return float(mae), len(X_test), {"eval_mae": float(mae), "model_used": "global_ensemble"}
        except Exception as e:
            self.logger.info(f"Errore evaluate: {e}")
            return float("nan"), len(X_test), {"eval_mae": float("nan")}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_app.py <client_id> [optional_data_path]")
        sys.exit(1)

    client_id = int(sys.argv[1])

    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    else:
        data_path = str(CLIENTS_DATA_DIR / f"group{client_id}_merged_clean.csv")

    if not os.path.exists(data_path):
        print(f"ERRORE: Data file not found: {data_path}")
        sys.exit(1)

    client = ExtraTreesClient(cid=client_id, data_path=data_path)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
