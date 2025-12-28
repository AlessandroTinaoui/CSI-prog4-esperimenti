import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


# -------------------------
# Helpers: bytes <-> Parameters
# -------------------------
def _params_from_bytes(blob: Optional[bytes]) -> Parameters:
    if not blob:
        return ndarrays_to_parameters([])
    arr = np.frombuffer(blob, dtype=np.uint8)
    return ndarrays_to_parameters([arr])


def _bytes_from_params(parameters: Parameters) -> Optional[bytes]:
    nds = parameters_to_ndarrays(parameters)
    if not nds:
        return None
    if nds[0].size == 0:
        return None
    return np.array(nds[0], dtype=np.uint8).tobytes()


@dataclass
class Stump:
    feature: str
    thr: float
    w_left: float
    w_right: float
    lr: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "thr": float(self.thr),
            "w_left": float(self.w_left),
            "w_right": float(self.w_right),
            "lr": float(self.lr),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Stump":
        return Stump(
            feature=str(d["feature"]),
            thr=float(d["thr"]),
            w_left=float(d["w_left"]),
            w_right=float(d["w_right"]),
            lr=float(d.get("lr", 1.0)),
        )


class FederatedHistStumpStrategy(FedAvg):
    """
    Round 1: feature selection (come nel tuo progetto)
    Round 2: binning globale (quantili) per le feature selezionate
    Round >=3: 1 stump per round usando istogrammi aggregati (G/H) dai client
    """

    def __init__(
        self,
        top_k: int = 30,
        n_bins: int = 64,
        huber_delta: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        learning_rate: float = 0.1,
        base_score: float = 0.0,
        save_path_features: str = "selected_features.json",
        save_path_model="global_model.json",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.top_k = int(top_k)
        self.n_bins = int(n_bins)
        self.huber_delta = float(huber_delta)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.base_score = float(base_score)

        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None

        # global bin edges: feature -> list[float] (len = n_bins+1)
        self.bin_edges: Optional[Dict[str, List[float]]] = None

        # model: list of stumps
        self.model: List[Stump] = []

        # results dir (come fai già)
        self._project_root = Path(__file__).resolve().parents[1]
        self._results_dir = self._project_root / "results"
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self._save_path_features = self._results_dir / Path(save_path_features).name
        self._save_path_model = self._results_dir / Path(save_path_model).name

    # -------------------------
    # Model serialization
    # -------------------------
    def _model_bytes(self) -> bytes:
        payload = {
            "base_score": self.base_score,
            "stumps": [s.to_dict() for s in self.model],
            "selected_features": self.selected_features,
            "bin_edges": self.bin_edges,
        }
        return json.dumps(payload).encode("utf-8")

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        # all'inizio modello vuoto
        return _params_from_bytes(self._model_bytes())

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)
        new_instructions = []

        if server_round == 1:
            # feature selection
            for cp, ins in fit_instructions:
                ins.config["phase"] = "fs"
                ins.config["top_k"] = str(self.top_k)
                ins.config["server_round"] = str(server_round)
                new_instructions.append((cp, ins))
            return new_instructions

        if server_round == 2:
            # binning globale: chiedi quantili locali per selected_features
            for cp, ins in fit_instructions:
                ins.config["phase"] = "binning"
                ins.config["n_bins"] = str(self.n_bins)
                ins.config["server_round"] = str(server_round)
                if self.selected_features is not None:
                    ins.config["selected_features"] = json.dumps(self.selected_features)
                new_instructions.append((cp, ins))
            return new_instructions

        # Round >=3: training stump
        for cp, ins in fit_instructions:
            ins.config["phase"] = "train"
            ins.config["server_round"] = str(server_round)
            ins.config["huber_delta"] = str(self.huber_delta)
            ins.config["reg_lambda"] = str(self.reg_lambda)
            ins.config["gamma"] = str(self.gamma)
            ins.config["learning_rate"] = str(self.learning_rate)
            if self.selected_features is not None:
                ins.config["selected_features"] = json.dumps(self.selected_features)
            if self.bin_edges is not None:
                ins.config["bin_edges"] = json.dumps(self.bin_edges)
            new_instructions.append((cp, ins))
        return new_instructions

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        eval_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        new_instructions = []
        for cp, ins in eval_instructions:
            ins.config["server_round"] = str(server_round)
            if self.selected_features is not None:
                ins.config["selected_features"] = json.dumps(self.selected_features)
            if self.bin_edges is not None:
                ins.config["bin_edges"] = json.dumps(self.bin_edges)
            new_instructions.append((cp, ins))
        return new_instructions

    # -------------------------
    # Aggregation logic
    # -------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return _params_from_bytes(self._model_bytes()), {}

        # ==============
        # ROUND 1: FS
        # ==============
        if server_round == 1:
            importances_list = []
            weights = []
            feature_names = None

            for _, fit_res in results:
                m = fit_res.metrics or {}
                if "local_feature_importance" not in m or "feature_names" not in m:
                    continue
                try:
                    local_imp = np.array(json.loads(m["local_feature_importance"]), dtype=float)
                    local_names = json.loads(m["feature_names"])
                except Exception:
                    continue

                if feature_names is None:
                    feature_names = local_names
                elif local_names != feature_names:
                    continue

                importances_list.append(local_imp)
                weights.append(float(fit_res.num_examples))

            if not importances_list or feature_names is None:
                self.selected_features = None
                return _params_from_bytes(self._model_bytes()), {"fs_done": 0.0}

            importances_mat = np.vstack(importances_list)
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
            agg_imp = (importances_mat * weights[:, None]).sum(axis=0)

            top_idx = np.argsort(-agg_imp)[: self.top_k]
            selected = [feature_names[i] for i in top_idx]

            self.feature_names = feature_names
            self.selected_features = selected
            # Salva le feature nel formato atteso dal vecchio server_flwr.py
            global_feat_path = self._results_dir / "global_model_features.json"
            with open(global_feat_path, "w", encoding="utf-8") as f:
                json.dump({"features": list(self.selected_features)}, f, ensure_ascii=False, indent=2)

            with open(self._save_path_features, "w", encoding="utf-8") as f:
                json.dump({"selected_features": selected}, f, ensure_ascii=False, indent=2)

            # reset model/binning (per sicurezza)
            self.model = []
            self.bin_edges = None

            return _params_from_bytes(self._model_bytes()), {
                "fs_done": 1.0,
                "n_selected_features": float(len(selected)),
            }

        # ==================
        # ROUND 2: BINNING
        # ==================
        if server_round == 2:
            # ogni client invia metrics["bin_edges"] come dict feature -> edges
            edges_by_feature: Dict[str, List[List[float]]] = {}
            for _, fit_res in results:
                m = fit_res.metrics or {}
                if "bin_edges" not in m:
                    continue
                try:
                    local_edges = json.loads(m["bin_edges"])
                except Exception:
                    continue
                if not isinstance(local_edges, dict):
                    continue
                for feat, edges in local_edges.items():
                    if not isinstance(edges, list):
                        continue
                    edges_by_feature.setdefault(feat, []).append([float(x) for x in edges])

            if not edges_by_feature:
                return _params_from_bytes(self._model_bytes()), {"binning_done": 0.0}

            # aggrega: mediana per indice
            global_edges: Dict[str, List[float]] = {}
            for feat, edge_lists in edges_by_feature.items():
                arr = np.array(edge_lists, dtype=float)  # shape (n_clients, n_bins+1)
                if arr.ndim != 2 or arr.shape[1] < 3:
                    continue
                med = np.median(arr, axis=0)
                # monotonic fix (quantili possono avere ripetizioni)
                med = np.maximum.accumulate(med)
                global_edges[feat] = med.tolist()

            self.bin_edges = global_edges

            # persist
            with open(self._save_path_model, "w", encoding="utf-8") as f:
                json.dump(json.loads(self._model_bytes().decode("utf-8")), f, ensure_ascii=False, indent=2)

            return _params_from_bytes(self._model_bytes()), {
                "binning_done": 1.0,
                "n_binned_features": float(len(global_edges)),
            }

        # ==========================
        # ROUND >= 3: BUILD 1 STUMP
        # ==========================
        # metric payload atteso: hist_g / hist_h / totals per feature
        # hist_g[feat] = [sum_g_bin0, ..., sum_g_bin{B-1}]
        # hist_h[feat] = [sum_h_bin0, ..., sum_h_bin{B-1}]
        agg_g: Dict[str, np.ndarray] = {}
        agg_h: Dict[str, np.ndarray] = {}
        total_G = 0.0
        total_H = 0.0

        total_examples = 0
        total_mae = 0.0

        for _, fit_res in results:
            m = fit_res.metrics or {}
            total_examples += fit_res.num_examples
            if "train_mae" in m:
                total_mae += float(m["train_mae"]) * float(fit_res.num_examples)

            try:
                local_total_G = float(m.get("total_G", 0.0))
                local_total_H = float(m.get("total_H", 0.0))
                total_G += local_total_G
                total_H += local_total_H

                hist_g = json.loads(m.get("hist_g", "{}"))
                hist_h = json.loads(m.get("hist_h", "{}"))
            except Exception:
                continue

            if not isinstance(hist_g, dict) or not isinstance(hist_h, dict):
                continue

            for feat, g_list in hist_g.items():
                if feat not in hist_h:
                    continue
                g_arr = np.array(g_list, dtype=float)
                h_arr = np.array(hist_h[feat], dtype=float)
                if g_arr.shape != h_arr.shape or g_arr.ndim != 1:
                    continue
                agg_g[feat] = agg_g.get(feat, np.zeros_like(g_arr)) + g_arr
                agg_h[feat] = agg_h.get(feat, np.zeros_like(h_arr)) + h_arr

        if not agg_g or self.bin_edges is None:
            return _params_from_bytes(self._model_bytes()), {"stump_added": 0.0}

        # funzione gain (XGBoost-like)
        lam = self.reg_lambda
        gamma = self.gamma

        def score(G: float, H: float) -> float:
            return (G * G) / (H + lam)

        best_feat = None
        best_bin = None
        best_gain = -1e18

        parent_score = score(total_G, total_H)

        for feat, g_bins in agg_g.items():
            h_bins = agg_h.get(feat)
            if h_bins is None:
                continue

            # prefix sums per split: left = bins <= t
            G_prefix = np.cumsum(g_bins)
            H_prefix = np.cumsum(h_bins)

            for b in range(len(g_bins) - 1):  # split tra b e b+1
                G_L = float(G_prefix[b])
                H_L = float(H_prefix[b])
                G_R = float(total_G - G_L)
                H_R = float(total_H - H_L)

                gain = score(G_L, H_L) + score(G_R, H_R) - parent_score - gamma
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_bin = b

        if best_feat is None or best_bin is None:
            return _params_from_bytes(self._model_bytes()), {"stump_added": 0.0}

        # threshold: usa edge tra bin e bin+1
        edges = self.bin_edges.get(best_feat)
        if not edges or best_bin + 1 >= len(edges):
            return _params_from_bytes(self._model_bytes()), {"stump_added": 0.0}

        thr = float(edges[best_bin + 1])

        # leaf weights: w = -G/(H+lambda)
        g_bins = agg_g[best_feat]
        h_bins = agg_h[best_feat]
        G_prefix = np.cumsum(g_bins)
        H_prefix = np.cumsum(h_bins)
        G_L = float(G_prefix[best_bin])
        H_L = float(H_prefix[best_bin])
        G_R = float(total_G - G_L)
        H_R = float(total_H - H_L)

        w_left = -G_L / (H_L + lam)
        w_right = -G_R / (H_R + lam)

        stump = Stump(
            feature=best_feat,
            thr=thr,
            w_left=w_left,
            w_right=w_right,
            lr=self.learning_rate,
        )
        self.model.append(stump)

        # persist model snapshot
        with open(self._save_path_model, "w", encoding="utf-8") as f:
            json.dump(json.loads(self._model_bytes().decode("utf-8")), f, ensure_ascii=False, indent=2)

        avg_train_mae = (total_mae / total_examples) if total_examples > 0 else float("nan")

        return _params_from_bytes(self._model_bytes()), {
            "stump_added": 1.0,
            "best_gain": float(best_gain),
            "best_feature": str(best_feat),
            "n_stumps": float(len(self.model)),
            "train_mae": float(avg_train_mae) if avg_train_mae == avg_train_mae else float("nan"),
        }

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_examples = 0
        total_mae = 0.0
        for _, ev in results:
            total_examples += ev.num_examples
            total_mae += float(ev.loss) * float(ev.num_examples)

        avg_mae = total_mae / total_examples if total_examples > 0 else float("nan")
        return float(avg_mae), {"eval_mae": float(avg_mae)}
