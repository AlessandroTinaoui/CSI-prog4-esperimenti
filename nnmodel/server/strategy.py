# strategy.py  (ADATTATA)

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import torch

from nnmodel.model import MLPRegressor

from nnmodel.server.config import (
    RESULTS_DIRNAME, GLOBAL_FEATURES_JSON, GLOBAL_SCALER_JSON,
    HIDDEN_SIZES, DROPOUT
)


class FedAvgNNWithGlobalScaler(FedAvg):
    def __init__(self, project_root: Path, **kwargs: Any):
        super().__init__(**kwargs)
        self.project_root = project_root
        self.results_dir = self.project_root / RESULTS_DIRNAME
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.global_features: Optional[List[str]] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)
        out = []

        if server_round == 1:
            for cp, ins in fit_instructions:
                ins.config["phase"] = "scaler"
                out.append((cp, ins))
            return out

        if self.global_features is None or self.scaler_mean is None or self.scaler_std is None:
            raise RuntimeError("Scaler non inizializzato: round 1 non ha prodotto mean/std.")

        for cp, ins in fit_instructions:
            ins.config["phase"] = "train"
            ins.config["global_features"] = json.dumps(self.global_features)
            ins.config["scaler_mean"] = json.dumps(self.scaler_mean.tolist())
            ins.config["scaler_std"] = json.dumps(self.scaler_std.tolist())
            out.append((cp, ins))
        return out

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        eval_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        out = []

        if server_round == 1:
            return []

        for cp, ins in eval_instructions:
            ins.config["global_features"] = json.dumps(self.global_features)
            ins.config["scaler_mean"] = json.dumps(self.scaler_mean.tolist())
            ins.config["scaler_std"] = json.dumps(self.scaler_std.tolist())
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
        # ROUND 1: build global scaler
        # -------------------------
        if server_round == 1:
            feats_union = set()
            per_client = []
            for _, fit_res in results:
                m = fit_res.metrics or {}
                if "feature_names" not in m:
                    continue
                feat_names = json.loads(m["feature_names"])
                feats_union.update(feat_names)
                per_client.append((feat_names, m))

            global_features = sorted(list(feats_union))
            d = len(global_features)
            if d == 0:
                raise RuntimeError("Nessuna feature ricevuta in round 1.")

            N = 0
            SUM = np.zeros(d, dtype=np.float64)
            SUMSQ = np.zeros(d, dtype=np.float64)

            idx = {f: i for i, f in enumerate(global_features)}

            for feat_names, m in per_client:
                n = int(m["n"])
                s = np.array(json.loads(m["sum"]), dtype=np.float64)
                ssq = np.array(json.loads(m["sumsq"]), dtype=np.float64)

                for j, f in enumerate(feat_names):
                    gi = idx[f]
                    SUM[gi] += s[j]
                    SUMSQ[gi] += ssq[j]
                N += n

            if N <= 1:
                raise RuntimeError("Troppi pochi esempi aggregati per calcolare std.")

            mean = SUM / N
            var = (SUMSQ / N) - (mean * mean)
            var = np.maximum(var, 1e-12)
            std = np.sqrt(var)

            self.global_features = global_features
            self.scaler_mean = mean.astype(np.float32)
            self.scaler_std = std.astype(np.float32)

            (self.results_dir / GLOBAL_FEATURES_JSON).write_text(
                json.dumps({"features": self.global_features}, indent=2),
                encoding="utf-8",
            )
            (self.results_dir / GLOBAL_SCALER_JSON).write_text(
                json.dumps(
                    {"mean": self.scaler_mean.tolist(), "std": self.scaler_std.tolist()},
                    indent=2,
                ),
                encoding="utf-8",
            )

            # ---- Crea pesi iniziali globali coerenti per il round 2 ----
            # (così tutti i client partono dagli stessi pesi, non random diversi)
            torch.manual_seed(0)  # deterministico
            init_model = MLPRegressor(
                input_dim=d,
                hidden_sizes=HIDDEN_SIZES,
                dropout=DROPOUT,
            )
            init_params = [val.detach().cpu().numpy() for _, val in init_model.state_dict().items()]

            return ndarrays_to_parameters(init_params), {
                "scaler_done": 1.0,
                "n_features": float(d),
                "N": float(N),
            }


        # -------------------------
        # ROUND >=2: normal FedAvg aggregation + SAVE MODEL
        # -------------------------
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # ✅ Salva sempre dal round 2 in poi (sovrascrive: avrai sempre l’ultimo modello)
        if aggregated_parameters is not None and server_round >= 2:
            nds = parameters_to_ndarrays(aggregated_parameters)
            if len(nds) > 0:
                out_path = self.results_dir / "global_model.npz"
                np.savez(out_path, *nds)
                print(f"[SERVER] Salvato global_model.npz in: {out_path.resolve()}")

        return aggregated_parameters, aggregated_metrics
