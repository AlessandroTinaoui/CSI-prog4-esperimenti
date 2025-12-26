from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from config import RESULTS_DIRNAME, GLOBAL_FEATURES_JSON, GLOBAL_SCALER_JSON
from TabNet.model import TabNetRegressor, TabNetConfig

import TabNet.client.client_params as P


class FedProxNNWithGlobalScaler(FedAvg):
    """
    Aggregazione: FedAvg (come prima)
    Training client-side: FedProx tramite penalità prox (mu) che inviamo ai client via config.
    """
    def __init__(self, project_root: Path, fedprox_mu: float = 0.01, **kwargs: Any):
        super().__init__(**kwargs)
        self.project_root = project_root
        self.results_dir = self.project_root / RESULTS_DIRNAME
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.fedprox_mu = float(fedprox_mu)

        self.global_features: Optional[List[str]] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None

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

            # FedProx: invia mu al client
            ins.config["fedprox_mu"] = str(self.fedprox_mu)

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

        # ROUND 1: global scaler + y stats
        if server_round == 1:
            feats_union = set()
            per_client = []

            YN = 0
            YSUM = 0.0
            YSUMSQ = 0.0

            for _, fit_res in results:
                m = fit_res.metrics or {}
                if "feature_names" not in m:
                    continue
                feat_names = json.loads(m["feature_names"])
                feats_union.update(feat_names)
                per_client.append((feat_names, m))

                if "y_n" in m and "y_sum" in m and "y_sumsq" in m:
                    YN += int(m["y_n"])
                    YSUM += float(m["y_sum"])
                    YSUMSQ += float(m["y_sumsq"])

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
                raise RuntimeError("Troppi pochi esempi aggregati per calcolare std (X).")

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
                json.dumps({"mean": self.scaler_mean.tolist(), "std": self.scaler_std.tolist()}, indent=2),
                encoding="utf-8",
            )

            if YN <= 1:
                self.y_mean = 0.0
                self.y_std = 1.0
            else:
                y_mean = YSUM / YN
                y_var = (YSUMSQ / YN) - (y_mean * y_mean)
                y_var = float(max(y_var, 1e-12))
                self.y_mean = float(y_mean)
                self.y_std = float(np.sqrt(y_var))

            (self.results_dir / "global_target.json").write_text(
                json.dumps({"y_mean": self.y_mean, "y_std": self.y_std}, indent=2),
                encoding="utf-8",
            )

            # init TabNet weights
            torch.manual_seed(0)
            tabcfg = TabNetConfig(
                n_d=P.TABNET_N_D,
                n_a=P.TABNET_N_A,
                n_steps=P.TABNET_N_STEPS,
                gamma=P.TABNET_GAMMA,
                n_shared=P.TABNET_N_SHARED,
                n_independent=P.TABNET_N_INDEPENDENT,
                bn_virtual_bs=P.TABNET_BN_VIRTUAL_BS,
                bn_momentum=P.TABNET_BN_MOMENTUM,
            )
            init_model = TabNetRegressor(input_dim=d, cfg=tabcfg)
            init_params = [val.detach().cpu().numpy() for _, val in init_model.state_dict().items()]

            return ndarrays_to_parameters(init_params), {
                "scaler_done": 1.0,
                "n_features": float(d),
                "N": float(N),
                "Y_N": float(YN),
            }

        # ROUND >=2: FedAvg aggregation + save model
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None and server_round >= 2:
            nds = parameters_to_ndarrays(aggregated_parameters)
            if len(nds) > 0:
                out_path = self.results_dir / "global_model.npz"
                np.savez(out_path, *nds)
                print(f"[SERVER] Salvato global_model.npz in: {out_path.resolve()}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            return aggregated_loss, aggregated_metrics

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

        return aggregated_loss, aggregated_metrics
