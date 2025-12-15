# strategy.py
import pickle
import json
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from flwr.common import Parameters, FitRes, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class RandomForestAggregation(FedAvg):
    """Strategia: Round1 feature selection (L1+L2), Round2+ aggregazione RandomForest combinando alberi."""

    def __init__(
        self,
        missing_threshold: float = 0.4,
        var_threshold: float = 1e-8,
        eps_l2: float = 0.0,
        min_clients_ratio: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.missing_threshold = missing_threshold
        self.var_threshold = var_threshold
        self.eps_l2 = eps_l2
        self.min_clients_ratio = min_clients_ratio
        self.selected_features: List[str] = []

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # Round 1: chiedi stats per feature selection
        if server_round == 1:
            config = {"task": "feature_select"}
        else:
            config = {
                "task": "train",
                "selected_features_json": json.dumps(self.selected_features),
            }
        return super().configure_fit(server_round, parameters, client_manager, config)

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        config = {"selected_features_json": json.dumps(self.selected_features)}
        return super().configure_evaluate(server_round, parameters, client_manager, config)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # -------------------
        # ROUND 1: FEATURE SELECTION (L1 + L2)
        # -------------------
        if server_round == 1:
            # raccogli da tutti i client
            all_features = set()
            l1_missing_list = []
            l1_var_list = []
            l1_n_list = []
            l2_imp_list = []

            for _, fit_res in results:
                n = int(fit_res.metrics.get("l1_n", 0))
                feats = json.loads(fit_res.metrics.get("l1_features_json", "[]"))
                miss = json.loads(fit_res.metrics.get("l1_missing_json", "{}"))
                var = json.loads(fit_res.metrics.get("l1_var_json", "{}"))
                imp = json.loads(fit_res.metrics.get("l2_importances_json", "{}"))

                all_features.update(feats)
                l1_missing_list.append(miss)
                l1_var_list.append(var)
                l1_n_list.append(max(n, 1))
                l2_imp_list.append(imp)

            all_features = sorted(all_features)

            # ---- L1 aggregation (media pesata per n)
            total_n = float(sum(l1_n_list))
            miss_global = {f: 0.0 for f in all_features}
            var_global = {f: 0.0 for f in all_features}

            for miss, var, n in zip(l1_missing_list, l1_var_list, l1_n_list):
                w = n / total_n
                for f in all_features:
                    miss_global[f] += w * float(miss.get(f, 0.0))
                    var_global[f] += w * float(var.get(f, 0.0))

            drop_l1 = set([f for f in all_features if miss_global[f] > self.missing_threshold])
            drop_l1 |= set([f for f in all_features if var_global[f] < self.var_threshold])

            # ---- L2 aggregation (mediana robusta)
            # matrice importanze [clients x features]
            mat = np.array([[float(imp.get(f, 0.0)) for f in all_features] for imp in l2_imp_list], dtype=float)
            med = np.median(mat, axis=0)
            frac_low = (mat <= self.eps_l2).mean(axis=0)

            drop_l2 = set()
            for f, m, flw in zip(all_features, med, frac_low):
                if (m <= self.eps_l2) and (flw >= self.min_clients_ratio):
                    drop_l2.add(f)

            # selezione finale: tieni tutto tranne drop_l1 e drop_l2
            self.selected_features = [f for f in all_features if f not in drop_l1 and f not in drop_l2]

            metrics = {
                "selected_features_count": len(self.selected_features),
                "dropped_l1_count": len(drop_l1),
                "dropped_l2_count": len(drop_l2),
            }
            print(f"[Server] Selected {len(self.selected_features)} features.")
            return parameters, metrics  # parameters invariati nel round1

        # -------------------
        # ROUND 2+: RF aggregation (come il tuo codice)
        # -------------------
        all_models = []
        total_examples = 0

        for _, fit_res in results:
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

        combined_model = all_models[0][0]
        all_estimators = []
        for model, _ in all_models:
            all_estimators.extend(model.estimators_)

        combined_model.estimators_ = all_estimators
        combined_model.n_estimators = len(all_estimators)

        model_bytes = pickle.dumps(combined_model)
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        parameters = ndarrays_to_parameters([model_array])

        total_acc = sum([fit_res.metrics.get("train_accuracy", 0) * fit_res.num_examples for _, fit_res in results])
        avg_accuracy = total_acc / total_examples if total_examples > 0 else 0.0

        metrics_aggregated = {
            "train_accuracy": float(avg_accuracy),
            "total_trees": int(combined_model.n_estimators),
        }
        return parameters, metrics_aggregated
