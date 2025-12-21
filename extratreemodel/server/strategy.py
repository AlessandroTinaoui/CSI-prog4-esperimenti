import json
import pickle
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


# ---------- Helpers: bytes <-> Flower Parameters (uint8 ndarray) ----------

def _params_from_bytes(model_bytes: Optional[bytes]) -> Parameters:
    if not model_bytes:
        return ndarrays_to_parameters([])
    arr = np.frombuffer(model_bytes, dtype=np.uint8)
    return ndarrays_to_parameters([arr])


def _bytes_from_params(parameters: Parameters) -> Optional[bytes]:
    nds = parameters_to_ndarrays(parameters)
    if not nds:
        return None
    if nds[0].size == 0:
        return None
    return np.array(nds[0], dtype=np.uint8).tobytes()


# ---------- Global Ensemble wrapper ----------

class ExtraTreesEnsemble:
    """
    Ensemble semplice: media delle predizioni di più ExtraTreesRegressor.
    Mantiene anche la lista delle feature selezionate per l'allineamento.
    """

    def __init__(self, models: List[Any], feature_names: List[str]):
        self.models = list(models)
        self.feature_names = list(feature_names)

    def predict(self, X):
        if len(self.models) == 0:
            raise ValueError("Ensemble vuoto: nessun modello disponibile.")
        preds = []
        for m in self.models:
            preds.append(m.predict(X))
        return np.mean(np.vstack(preds), axis=0)


class ExtraTreesEnsembleStrategy(FedAvg):
    """
    FL con ExtraTrees:
      - Round 1: federated feature selection (ogni client invia feature_importances_)
      - Round >=2: ogni client invia un modello ExtraTrees completo (pickle).
                  Il server crea un ensemble e lo rimanda ai client per evaluate.
    """

    def __init__(
        self,
        top_k: int = 20,
        save_path: str = "selected_features.json",
        global_model_path: str = "global_model.pkl",
        max_models_in_ensemble: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.top_k = int(top_k)
        self.max_models_in_ensemble = max_models_in_ensemble

        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None

        self._global_ensemble: Optional[ExtraTreesEnsemble] = None
        self._global_model_bytes: Optional[bytes] = None

        self._project_root = Path(__file__).resolve().parents[1]
        self._results_dir = self._project_root / "results"
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self._save_path_abs = self._results_dir / Path(save_path).name
        self._global_model_path = self._results_dir / Path(global_model_path).name
        self._global_features_path = self._results_dir / "global_model_features.json"

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        # Round 1 non manda modello (FS), quindi inizializza vuoto.
        if self._global_model_bytes:
            return _params_from_bytes(self._global_model_bytes)
        return ndarrays_to_parameters([])

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)
        new_instructions = []

        if server_round == 1:
            for cp, fit_ins in fit_instructions:
                fit_ins.config["phase"] = "fs"
                fit_ins.config["top_k"] = str(self.top_k)
                fit_ins.config["server_round"] = str(server_round)
                new_instructions.append((cp, fit_ins))
            return new_instructions

        # Round >=2: training
        for cp, fit_ins in fit_instructions:
            fit_ins.config["phase"] = "train"
            if self.selected_features is not None:
                fit_ins.config["selected_features"] = json.dumps(self.selected_features)
            fit_ins.config["server_round"] = str(server_round)
            new_instructions.append((cp, fit_ins))
        return new_instructions

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        eval_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        new_instructions = []
        for cp, eval_ins in eval_instructions:
            if self.selected_features is not None:
                eval_ins.config["selected_features"] = json.dumps(self.selected_features)
            eval_ins.config["server_round"] = str(server_round)
            new_instructions.append((cp, eval_ins))
        return new_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # -------------------------
        # ROUND 1: FEATURE SELECTION
        # -------------------------
        if server_round == 1:
            print(f"[SERVER] Round 1: Feature Selection su {len(results)} client...")

            importances_list = []
            weights = []
            feature_names = None

            for _, fit_res in results:
                if not fit_res.metrics:
                    continue
                n = fit_res.num_examples
                m = fit_res.metrics
                if "local_feature_importance" not in m or "feature_names" not in m:
                    continue

                try:
                    local_imp = np.array(json.loads(m["local_feature_importance"]), dtype=float)
                    local_names = json.loads(m["feature_names"])
                except (json.JSONDecodeError, TypeError):
                    continue

                if feature_names is None:
                    feature_names = local_names
                elif local_names != feature_names:
                    # se un client ha colonne diverse, lo saltiamo per stabilità
                    continue

                importances_list.append(local_imp)
                weights.append(n)

            if not importances_list or feature_names is None:
                print("⚠️ Nessuna feature importance ricevuta. Salto la selezione.")
                self.feature_names = None
                self.selected_features = None
                return ndarrays_to_parameters([]), {"fs_done": 0.0}

            importances_mat = np.vstack(importances_list)
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
            agg_imp = (importances_mat * weights[:, None]).sum(axis=0)

            top_idx = np.argsort(-agg_imp)[: self.top_k]
            selected = [feature_names[i] for i in top_idx]

            self.feature_names = feature_names
            self.selected_features = selected

            # salva selected_features.json
            with open(self._save_path_abs, "w", encoding="utf-8") as f:
                json.dump({"selected_features": selected}, f, ensure_ascii=False, indent=2)

            # salva global_model_features.json (coerente col tuo server_flwr)
            with open(self._global_features_path, "w", encoding="utf-8") as f:
                json.dump({"features": list(self.selected_features)}, f, ensure_ascii=False, indent=2)

            print(f"[SERVER] Feature salvate: {self._global_features_path}")

            return ndarrays_to_parameters([]), {
                "fs_done": 1.0,
                "n_selected_features": float(len(selected)),
            }

        # -------------------------
        # ROUND >=2: BUILD ENSEMBLE
        # -------------------------
        models: List[Any] = []
        total_examples = 0
        total_train_mae = 0.0

        for _, fit_res in results:
            total_examples += fit_res.num_examples
            if fit_res.metrics and "train_mae" in fit_res.metrics:
                total_train_mae += float(fit_res.metrics["train_mae"]) * fit_res.num_examples

            local_bytes = _bytes_from_params(fit_res.parameters)
            if not local_bytes:
                continue

            try:
                local_model = pickle.loads(local_bytes)
                models.append(local_model)
            except Exception:
                print("⚠️ Impossibile deserializzare modello client (pickle).")
                continue

        if self.selected_features is None:
            # fallback: se per qualche motivo non abbiamo selezionato, usiamo feature_names
            if self.feature_names is None:
                raise RuntimeError("Nessuna feature disponibile (né selected né feature_names).")
            feats = self.feature_names
        else:
            feats = self.selected_features

        if self.max_models_in_ensemble is not None and len(models) > self.max_models_in_ensemble:
            # tieni gli ultimi N ricevuti (semplice e stabile)
            models = models[-int(self.max_models_in_ensemble):]

        self._global_ensemble = ExtraTreesEnsemble(models=models, feature_names=feats)
        self._global_model_bytes = pickle.dumps(self._global_ensemble)

        # salva global_model.pkl
        with open(self._global_model_path, "wb") as f:
            f.write(self._global_model_bytes)

        avg_mae = total_train_mae / total_examples if total_examples > 0 else float("nan")

        metrics_aggregated: Dict[str, Scalar] = {
            "train_mae": float(avg_mae) if avg_mae == avg_mae else float("nan"),
            "n_models_in_ensemble": float(len(models)),
        }
        if self.selected_features is not None:
            metrics_aggregated["n_selected_features"] = float(len(self.selected_features))

        # IMPORTANT: ritorniamo l'ensemble come parameters, così i client lo ricevono per evaluate / round successivo
        return _params_from_bytes(self._global_model_bytes), metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_examples = 0
        total_loss = 0.0

        for _, eval_res in results:
            total_examples += eval_res.num_examples
            total_loss += float(eval_res.loss) * eval_res.num_examples

        avg_loss = total_loss / total_examples if total_examples > 0 else float("nan")
        return float(avg_loss), {"eval_loss": float(avg_loss)}
