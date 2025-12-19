import json
import pickle
from pathlib import Path
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


class XGBoostEnsembleAggregation(FedAvg):
    """
    Strategia FL per Random Forest:
    - Round 1: Federated Feature Selection (aggrega feature importance e decide selected_features)
    - Round >=2: Aggregazione Random Forest combinando gli alberi (come il tuo codice originale)
    """

    def __init__(
        self,
        top_k: int = 20,
        save_path: str = "selected_features.json",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.save_path = save_path

        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None

    # -------------------------
    # 1) CONFIGURE_FIT
    # - Round 1: FS
    # - Round >=2: TRAIN
    # -------------------------
    def configure_fit(self, server_round, parameters, client_manager):
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)

        # fit_instructions: List[Tuple[ClientProxy, FitIns]]
        new_instructions = []

        if server_round == 1:
            for client_proxy, fit_ins in fit_instructions:
                fit_ins.config["phase"] = "fs"
                fit_ins.config["top_k"] = str(self.top_k)
                # >>> FIX: passa anche il server_round al client <<<
                fit_ins.config["server_round"] = str(server_round)
                new_instructions.append((client_proxy, fit_ins))

            return new_instructions

        # Round >= 2: training con feature fissate
        for client_proxy, fit_ins in fit_instructions:
            fit_ins.config["phase"] = "train"
            if self.selected_features is not None:
                fit_ins.config["selected_features"] = json.dumps(self.selected_features)
            # >>> FIX: passa anche il server_round al client <<<
            fit_ins.config["server_round"] = str(server_round)
            new_instructions.append((client_proxy, fit_ins))

        return new_instructions

    # -------------------------
    # 2) AGGREGATE_FIT
    # - Round 1: aggrega importanze e decide features
    # - Round >=2: aggrega i modelli combinando gli alberi (tuo metodo)
    # -------------------------
    # -------------------------
    # 2) AGGREGATE_FIT
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
            print(f"[SERVER] Round 1: Feature Selection su {len(results)} client...")

            importances_list = []
            weights = []
            feature_names = None
            total_examples = 0

            # --- 1. ESTRAZIONE DATI DAI CLIENT ---
            for _, fit_res in results:
                # Se il client ha fallito o non ha mandato metriche, saltiamo
                if not fit_res.metrics:
                    continue

                n = fit_res.num_examples
                total_examples += n
                m = fit_res.metrics

                # Controlliamo che ci siano le chiavi giuste
                if "local_feature_importance" not in m or "feature_names" not in m:
                    continue

                # Deserializziamo i JSON inviati dal client
                try:
                    local_imp = np.array(json.loads(m["local_feature_importance"]), dtype=float)
                    local_names = json.loads(m["feature_names"])
                except (json.JSONDecodeError, TypeError):
                    continue

                # Al primo giro, salviamo i nomi delle feature
                if feature_names is None:
                    feature_names = local_names
                else:
                    # Controllo di sicurezza: i nomi devono coincidere
                    if local_names != feature_names:
                        # Se differiscono, per ora ignoriamo questo client per evitare crash
                        continue

                importances_list.append(local_imp)
                weights.append(n)

            # --- 2. CALCOLO IMPORTANZA GLOBALE ---
            if not importances_list or feature_names is None:
                print("ATTENZIONE: Nessuna feature importance ricevuta. Salto la selezione.")
                self.feature_names = None
                self.selected_features = None
                return ndarrays_to_parameters([]), {"fs_done": 0.0}


            # Stack delle importanze (num_clients x num_features)
            importances_mat = np.vstack(importances_list)

            # Normalizzazione pesi (in base al numero di esempi)
            weights = np.array(weights, dtype=float)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # Fallback se pesi zero (improbabile)
                weights = np.ones(len(weights)) / len(weights)

            # Media pesata delle importanze
            agg_imp = (importances_mat * weights[:, None]).sum(axis=0)

            # --- 3. SELEZIONE TOP-K ---
            # Indici delle feature ordinate per importanza decrescente
            top_idx = np.argsort(-agg_imp)[: self.top_k]
            selected = [feature_names[i] for i in top_idx]

            self.feature_names = feature_names
            self.selected_features = selected

            print(f"✅ Selezionate {len(selected)} feature su {len(feature_names)} disponibili.")

            # --- 4. SALVATAGGIO SU DISCO (FIX PERCORSI) ---
            # Usiamo il percorso del file corrente per essere sicuri
            STRATEGY_DIR = Path(__file__).resolve().parent

            # Salva selected_features.json (usato dalla strategy)
            save_path_abs = STRATEGY_DIR / self.save_path
            with open(save_path_abs, "w", encoding="utf-8") as f:
                json.dump({"selected_features": selected}, f, ensure_ascii=False, indent=2)

            # Salva global_model_features.json (usato dal server per il test finale)
            global_feat_path = STRATEGY_DIR / "global_model_features.json"
            with open(global_feat_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"features": list(self.selected_features)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"[SERVER] File salvati: {global_feat_path}")

            metrics_aggregated: Dict[str, Scalar] = {
                "fs_done": 1.0,
                "n_selected_features": float(len(selected)),
            }
            # Round 1 non ritorna parametri modello, solo metriche
            return ndarrays_to_parameters([]), metrics_aggregated

        # =========================
        # ROUND >=2: ENSEMBLE AGGREGATION (XGBoost boosters)
        # =========================
        ensemble_items: List[Dict[str, Any]] = []
        total_examples = 0
        total_train_mae = 0.0

        for _, fit_res in results:
            num_examples = fit_res.num_examples
            total_examples += num_examples

            # Accumula MAE per logging
            if fit_res.metrics and "train_mae" in fit_res.metrics:
                total_train_mae += float(fit_res.metrics["train_mae"]) * num_examples

            params = parameters_to_ndarrays(fit_res.parameters)
            if len(params) == 0 or params[0].size == 0:
                continue

            # Recupera il modello raw (bytes)
            raw_bytes = np.array(params[0], dtype=np.uint8).tobytes()
            ensemble_items.append({"raw": raw_bytes, "weight": int(num_examples)})

        if not ensemble_items:
            print("⚠️ Nessun modello ricevuto in questo round.")
            return ndarrays_to_parameters([]), {"ensemble_size": 0.0}

        # Limitiamo la dimensione dell'ensemble se necessario
        MAX_GLOBAL_BOOSTERS = 1000
        if len(ensemble_items) > MAX_GLOBAL_BOOSTERS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(ensemble_items), size=MAX_GLOBAL_BOOSTERS, replace=False)
            ensemble_items = [ensemble_items[i] for i in idx]

        # --- SALVA ENSEMBLE SU DISCO ---
        STRATEGY_DIR = Path(__file__).resolve().parent
        GLOBAL_ENSEMBLE_PATH = STRATEGY_DIR / "global_ensemble.pkl"

        with open(GLOBAL_ENSEMBLE_PATH, "wb") as f:
            pickle.dump(ensemble_items, f)

        avg_mae = total_train_mae / total_examples if total_examples > 0 else 0.0

        metrics_aggregated: Dict[str, Scalar] = {
            "train_mae": float(avg_mae),
            "ensemble_size": float(len(ensemble_items)),
        }

        if self.selected_features is not None:
            metrics_aggregated["n_selected_features"] = float(len(self.selected_features))

        # Ritorniamo parametri vuoti perché l'aggregazione è salvata su disco (ensemble)
        # Flower richiede comunque un oggetto Parameters
        return ndarrays_to_parameters([]), metrics_aggregated
    # -------------------------
    # 3) AGGREGATE_EVALUATE
    # -------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        if not results:
            return ndarrays_to_parameters([]), {}

        total_examples = 0
        total_eval_mae = 0.0

        for client_proxy, fit_res in results:
            num_examples = fit_res.num_examples
            total_examples += num_examples
            eval_mae = float(fit_res.metrics.get("eval_mae", 0.0))
            total_eval_mae += eval_mae * num_examples

        avg_eval_mae = total_eval_mae / total_examples if total_examples > 0 else 0.0

        metrics_aggregated = {"eval_mae": float(avg_eval_mae)}
        return ndarrays_to_parameters([]), metrics_aggregated
