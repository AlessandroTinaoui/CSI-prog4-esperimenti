import json
import pickle
from typing import List, Tuple, Dict, Optional, Union, Any

import numpy as np
from flwr.common import (
    Parameters,
    FitRes,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class RandomForestAggregation(FedAvg):
    """
    Strategia FL per Random Forest:
    - Round 1: Federated Feature Selection (aggrega feature importance e decide selected_features)
    - Round >=2: Aggregazione Random Forest combinando gli alberi (come il tuo codice originale)
    """

    def __init__(
        self,
        top_k: int = 15,
        save_path: str = "selected_features.json",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.save_path = save_path

        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None

    # -------------------------
    # 1) CONFIGURE_FIT: manda phase e (dal round 2) selected_features
    # -------------------------
    def configure_fit(self, server_round, parameters, client_manager):
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)

        # fit_instructions: List[Tuple[ClientProxy, FitIns]]
        new_instructions = []

        if server_round == 1:
            for client_proxy, fit_ins in fit_instructions:
                fit_ins.config["phase"] = "fs"
                fit_ins.config["top_k"] = str(self.top_k)
                new_instructions.append((client_proxy, fit_ins))
            return new_instructions

        # Round >= 2: training con feature fissate
        for client_proxy, fit_ins in fit_instructions:
            fit_ins.config["phase"] = "train"
            if self.selected_features is not None:
                fit_ins.config["selected_features"] = json.dumps(self.selected_features)
            new_instructions.append((client_proxy, fit_ins))

        return new_instructions

    # -------------------------
    # 2) AGGREGATE_FIT
    # - Round 1: aggrega importanze e decide features
    # - Round >=2: aggrega i modelli combinando gli alberi (tuo metodo)
    # -------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # =========================
        # ROUND 1: FEATURE SELECTION
        # =========================
        if server_round == 1:
            importances_list = []
            weights = []
            feature_names = None
            total_examples = 0

            for _, fit_res in results:
                m = fit_res.metrics or {}
                n = fit_res.num_examples
                total_examples += n

                if "local_feature_importance" not in m or "feature_names" not in m:
                    continue

                local_imp = np.array(m["local_feature_importance"], dtype=float)
                local_names = json.loads(m["feature_names"])

                if feature_names is None:
                    feature_names = local_names
                else:
                    # Se per qualche motivo i nomi non coincidono, qui potresti gestire mismatch
                    # (per ora assumiamo che siano uguali)
                    pass

                importances_list.append(local_imp)
                weights.append(n)

            if not importances_list or feature_names is None:
                # Se non arrivano importanze, fallback: non seleziono nulla
                self.feature_names = None
                self.selected_features = None
                return None, {"fs_status": 0.0}

            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()

            agg_imp = np.zeros_like(importances_list[0], dtype=float)
            for imp, w in zip(importances_list, weights):
                agg_imp += imp * w

            # Top-k
            k = min(self.top_k, len(feature_names))
            idx = np.argsort(agg_imp)[::-1][:k]

            self.feature_names = feature_names
            self.selected_features = [feature_names[i] for i in idx]

            # salva
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"top_k": k, "selected_features": self.selected_features},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # Non serve mandare parameters nel round 1 (non stiamo aggregando modelli)
            metrics = {
                "fs_status": 1.0,
                "top_k": float(k),
                "total_examples_fs": float(total_examples),
            }
            return None, metrics

        # =========================
        # ROUND >=2: MODEL AGGREGATION (tuo codice originale, sistemato)
        # =========================
        all_models = []
        total_examples = 0

        for client_proxy, fit_res in results:
            num_examples = fit_res.num_examples
            total_examples += num_examples

            params = parameters_to_ndarrays(fit_res.parameters)

            if len(params) > 0 and params[0].size > 0:
                try:
                    model_bytes = params[0].tobytes()
                    model = pickle.loads(model_bytes)
                    all_models.append((model, num_examples))
                except Exception:
                    continue

        if not all_models:
            return None, {}

        # Combina gli alberi
        combined_model = all_models[0][0]
        all_estimators = []
        for model, _ in all_models:
            all_estimators.extend(model.estimators_)

        combined_model.estimators_ = all_estimators
        combined_model.n_estimators = len(all_estimators)


        # --- SALVA MODELLO GLOBALE SU DISCO (per test finale lato server) ---
        with open("global_model.pkl", "wb") as f:
            pickle.dump(combined_model, f)

        # --- SALVA LA LISTA DI FEATURE USATE IN TRAIN ---
        # (il modello sklearn dopo fit ha feature_names_in_ se X era un DataFrame)
        if hasattr(combined_model, "feature_names_in_"):
            with open("global_model_features.json", "w", encoding="utf-8") as f:
                json.dump({"features": combined_model.feature_names_in_.tolist()}, f, ensure_ascii=False, indent=2)

        # Serializza modello aggregato
        model_bytes = pickle.dumps(combined_model)
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        parameters = ndarrays_to_parameters([model_array])

        # Aggrega metriche training
        total_mae = 0.0
        for _, fit_res in results:
            total_mae += float(fit_res.metrics.get("train_mae", 0.0)) * fit_res.num_examples

        avg_mae = total_mae / total_examples if total_examples > 0 else 0.0

        metrics_aggregated: Dict[str, Scalar] = {
            "train_mae": float(avg_mae),
            "total_trees": float(combined_model.n_estimators),
        }

        # aggiungo info su feature selection
        if self.selected_features is not None:
            metrics_aggregated["n_selected_features"] = float(len(self.selected_features))

        return parameters, metrics_aggregated

    # -------------------------
    # 3) AGGREGATE_EVALUATE (fix: firma corretta)
    # -------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        total_examples = 0
        total_eval_mae = 0.0

        for client_proxy, fit_res in results:
            num_examples = fit_res.num_examples
            total_examples += num_examples
            eval_mae = float(fit_res.metrics.get("eval_mae", 0.0))
            total_eval_mae += eval_mae * num_examples

        avg_eval_mae = total_eval_mae / total_examples if total_examples > 0 else 0.0

        metrics_aggregated = {"eval_mae": float(avg_eval_mae)}
        # loss (float) può essere None se non lo usi
        return None, metrics_aggregated
