# strategy.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import torch

from mlp.model import MLPRegressor
from mlp.server.config import (
    RESULTS_DIRNAME,
    GLOBAL_FEATURES_JSON,
    GLOBAL_SCALER_JSON,
)

# Feature selection dal config (se non esistono, fallback sicuro)
try:
    from mlp.server.config import TOP_K_FEATURES, MIN_STD_FEATURE_FS
except Exception:
    TOP_K_FEATURES = 0
    MIN_STD_FEATURE_FS = 1e-8

from mlp.client.client_params import HIDDEN_SIZES, DROPOUT


class FedAvgNNWithGlobalScaler(FedAvg):
    def __init__(self, project_root: Path, **kwargs: Any):
        super().__init__(**kwargs)
        self.project_root = project_root
        self.results_dir = self.project_root / RESULTS_DIRNAME
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.global_features: Optional[List[str]] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

        # ✅ y global mean/std
        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None

        # ✅ tracking best global eval (lower is better)
        self.best_eval_mae: float = float("inf")
        self.best_round: int = -1

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)
        out = []

        if server_round == 1:
            for cp, ins in fit_instructions:
                ins.config["phase"] = "scaler"
                out.append((cp, ins))
            return out

        if (
            self.global_features is None
            or self.scaler_mean is None
            or self.scaler_std is None
            or self.y_mean is None
            or self.y_std is None
        ):
            raise RuntimeError("Artifacts non inizializzati: round 1 non ha prodotto scaler/target stats.")

        for cp, ins in fit_instructions:
            ins.config["phase"] = "train"
            ins.config["global_features"] = json.dumps(self.global_features)
            ins.config["scaler_mean"] = json.dumps(self.scaler_mean.tolist())
            ins.config["scaler_std"] = json.dumps(self.scaler_std.tolist())
            ins.config["y_mean"] = str(self.y_mean)
            ins.config["y_std"] = str(self.y_std)
            out.append((cp, ins))
        return out

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        eval_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        out = []

        if server_round == 1:
            return []

        if (
            self.global_features is None
            or self.scaler_mean is None
            or self.scaler_std is None
            or self.y_mean is None
            or self.y_std is None
        ):
            return []

        for cp, ins in eval_instructions:
            ins.config["global_features"] = json.dumps(self.global_features)
            ins.config["scaler_mean"] = json.dumps(self.scaler_mean.tolist())
            ins.config["scaler_std"] = json.dumps(self.scaler_std.tolist())
            ins.config["y_mean"] = str(self.y_mean)
            ins.config["y_std"] = str(self.y_std)
            out.append((cp, ins))
        return out

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # -------------------------
        # ROUND 1: build global scaler (X) + target stats (Y) + feature selection
        # -------------------------
        if server_round == 1:
            feats_union = set()
            per_client = []

            # accumulo anche y stats federate
            YN = 0
            YSUM = 0.0
            YSUMSQ = 0.0

            # opzionale: sum_xy per correlazione
            has_sumxy = False

            for _, fit_res in results:
                m = fit_res.metrics or {}

                if "feature_names" not in m:
                    continue

                feat_names = json.loads(m["feature_names"])
                feats_union.update(feat_names)
                per_client.append((feat_names, m))

                # y stats (se presenti)
                if "y_n" in m and "y_sum" in m and "y_sumsq" in m:
                    yn = int(m["y_n"])
                    ysum = float(m["y_sum"])
                    ysumsq = float(m["y_sumsq"])
                    YN += yn
                    YSUM += ysum
                    YSUMSQ += ysumsq

                if "sum_xy" in m:
                    has_sumxy = True

            global_features = sorted(list(feats_union))
            d = len(global_features)
            if d == 0:
                raise RuntimeError("Nessuna feature ricevuta in round 1.")

            # global scaler X
            N = 0
            SUM = np.zeros(d, dtype=np.float64)
            SUMSQ = np.zeros(d, dtype=np.float64)
            SUMXY = np.zeros(d, dtype=np.float64)  # se disponibile

            idx = {f: i for i, f in enumerate(global_features)}

            for feat_names, m in per_client:
                n = int(m["n"])
                s = np.array(json.loads(m["sum"]), dtype=np.float64)
                ssq = np.array(json.loads(m["sumsq"]), dtype=np.float64)

                sum_xy = None
                if "sum_xy" in m:
                    try:
                        sum_xy = np.array(json.loads(m["sum_xy"]), dtype=np.float64)
                    except Exception:
                        sum_xy = None

                for j, f in enumerate(feat_names):
                    gi = idx[f]
                    SUM[gi] += s[j]
                    SUMSQ[gi] += ssq[j]
                    if sum_xy is not None and j < sum_xy.shape[0]:
                        SUMXY[gi] += sum_xy[j]
                N += n

            if N <= 1:
                raise RuntimeError("Troppi pochi esempi aggregati per calcolare std (X).")

            mean = SUM / N
            var = (SUMSQ / N) - (mean * mean)
            var = np.maximum(var, 1e-12)
            std = np.sqrt(var)

            # ✅ global target y mean/std (federato)
            if YN <= 1:
                self.y_mean = 0.0
                self.y_std = 1.0
            else:
                y_mean = YSUM / YN
                y_var = (YSUMSQ / YN) - (y_mean * y_mean)
                y_var = float(max(y_var, 1e-12))
                self.y_mean = float(y_mean)
                self.y_std = float(np.sqrt(y_var))

            # -------------------------
            # FEATURE SELECTION (round 1)
            # 1) variance filter (std >= MIN_STD_FEATURE_FS)
            # 2) top-k per |corr(x,y)| se sum_xy presente
            # -------------------------
            top_k = int(TOP_K_FEATURES) if TOP_K_FEATURES is not None else 0
            min_std = float(MIN_STD_FEATURE_FS) if MIN_STD_FEATURE_FS is not None else 1e-8

            keep_mask = std >= min_std
            if not np.all(keep_mask):
                kept = int(np.sum(keep_mask))
                print(f"[SERVER] Round 1 - Variance filter: kept {kept}/{d} features (std >= {min_std})")

            global_features = [f for f, k in zip(global_features, keep_mask.tolist()) if k]
            mean = mean[keep_mask]
            std = std[keep_mask]
            SUMXY = SUMXY[keep_mask]

            # Top-k su correlazione (solo se disponibile)
            if top_k > 0:
                if not has_sumxy:
                    print("[SERVER] Round 1 - TOP_K_FEATURES > 0 ma 'sum_xy' non presente: salto top-k corr.")
                elif self.y_std is None or self.y_std <= 1e-12:
                    print("[SERVER] Round 1 - y_std ~ 0: salto top-k corr.")
                else:
                    d2 = len(global_features)
                    if d2 > 0:
                        exy = SUMXY / float(N)
                        cov = exy - (mean * float(self.y_mean))
                        denom = np.maximum(std, min_std) * float(self.y_std)
                        corr = cov / denom

                        k = min(top_k, d2)
                        order = np.argsort(-np.abs(corr))[:k]
                        order = np.sort(order)  # stabile

                        global_features = [global_features[i] for i in order.tolist()]
                        mean = mean[order]
                        std = std[order]
                        print(f"[SERVER] Round 1 - Top-k corr: kept {len(global_features)}/{d2} (k={k})")

            # salva artifacts finali
            self.global_features = global_features
            self.scaler_mean = mean.astype(np.float32)
            self.scaler_std = std.astype(np.float32)

            (self.results_dir / GLOBAL_FEATURES_JSON).write_text(
                json.dumps({"features": self.global_features}, indent=2),
                encoding="utf-8",
            )
            (self.results_dir / GLOBAL_SCALER_JSON).write_text(
                json.dumps({"mean": self.scaler_mean.tolist(), "std": self.scaler_std.tolist()}, indent=2),
                encoding="utf-8",
            )

            # salva anche su file per server_flwr.py
            (self.results_dir / "global_target.json").write_text(
                json.dumps({"y_mean": self.y_mean, "y_std": self.y_std}, indent=2),
                encoding="utf-8",
            )

            # ---- pesi iniziali coerenti per round 2 ----
            d_final = len(self.global_features)
            torch.manual_seed(0)
            init_model = MLPRegressor(
                input_dim=d_final,
                hidden_sizes=HIDDEN_SIZES,
                dropout=DROPOUT,
            )
            init_params = [val.detach().cpu().numpy() for _, val in init_model.state_dict().items()]

            return ndarrays_to_parameters(init_params), {
                "scaler_done": 1.0,
                "n_features": float(d_final),
                "N": float(N),
                "Y_N": float(YN),
                "y_mean": float(self.y_mean),
                "y_std": float(self.y_std),
            }

        # -------------------------
        # ROUND >=2: normal FedAvg + SAVE MODEL
        # -------------------------
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None and server_round >= 2:
            nds = parameters_to_ndarrays(aggregated_parameters)
            if len(nds) > 0:
                out_path = self.results_dir / "global_model.npz"
                np.savez(out_path, *nds)
                print(f"[SERVER] Salvato global_model.npz in: {out_path.resolve()}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """
        Aggrega i risultati di evaluate() dai client e salva il best model
        in base alla MAE reale (eval_mae_real) pesata per num_examples.
        """
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            return aggregated_loss, aggregated_metrics

        # Calcolo MAE federata pesata
        total_n = 0
        weighted_mae = 0.0

        for _, ev_res in results:
            m = ev_res.metrics or {}
            if "eval_mae_real" not in m:
                continue
            n = int(ev_res.num_examples)
            total_n += n
            weighted_mae += float(m["eval_mae_real"]) * n

        if total_n > 0:
            mean_mae = weighted_mae / total_n
            print(f"[SERVER] Round {server_round} - FED_EVAL_MAE_REAL: {mean_mae:.4f}")

            # Se migliora, copia global_model.npz -> best_model.npz
            if mean_mae < self.best_eval_mae:
                self.best_eval_mae = float(mean_mae)
                self.best_round = int(server_round)

                src = self.results_dir / "global_model.npz"
                dst = self.results_dir / "best_model.npz"
                if src.exists():
                    import shutil

                    shutil.copyfile(src, dst)
                    print(f"[SERVER] ✅ Best model aggiornato (round {server_round}) -> {dst.resolve()}")

            aggregated_metrics = aggregated_metrics or {}
            aggregated_metrics["fed_eval_mae_real"] = float(mean_mae)
            aggregated_metrics["best_eval_mae_real"] = float(self.best_eval_mae)
            aggregated_metrics["best_round"] = float(self.best_round)

        return aggregated_loss, aggregated_metrics
