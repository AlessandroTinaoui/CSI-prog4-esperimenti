import json
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


def _params_from_bytes(model_bytes: Optional[bytes]) -> Parameters:
    """Encode model bytes into Flower Parameters (as a single uint8 ndarray)."""
    if not model_bytes:
        return ndarrays_to_parameters([])
    arr = np.frombuffer(model_bytes, dtype=np.uint8)
    return ndarrays_to_parameters([arr])


def _bytes_from_params(parameters: Parameters) -> Optional[bytes]:
    """Decode model bytes from Flower Parameters."""
    nds = parameters_to_ndarrays(parameters)
    if not nds:
        return None
    if nds[0].size == 0:
        return None
    return np.array(nds[0], dtype=np.uint8).tobytes()


def _get_gbtree_section(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    return model_dict["learner"]["gradient_booster"]["model"]


def _get_num_parallel_tree(gbtree: Dict[str, Any]) -> int:
    p = gbtree.get("gbtree_model_param", {})
    try:
        return int(p.get("num_parallel_tree", 1))
    except Exception:
        return 1


def _recompute_iteration_indptr(num_trees: int, num_parallel_tree: int) -> List[int]:
    # iteration_indptr points to the start index of each boosting iteration in the flat tree list
    if num_parallel_tree <= 1:
        return list(range(0, num_trees + 1))
    n_iters = (num_trees + num_parallel_tree - 1) // num_parallel_tree
    indptr = [i * num_parallel_tree for i in range(0, n_iters + 1)]
    indptr[-1] = num_trees
    return indptr


def _set_tree_id(tree: Dict[str, Any], new_id: int) -> None:
    tree["id"] = new_id
    # Some XGBoost JSON versions also store the id inside tree_param
    if "tree_param" in tree and isinstance(tree["tree_param"], dict):
        tree["tree_param"]["id"] = str(new_id)


def _append_new_trees(global_dict: Dict[str, Any], local_dict: Dict[str, Any]) -> int:
    """
    Append to global_dict the trees that are present in local_dict but not yet in global_dict.

    Assumes local models are trained *starting from* the current global model
    (so local_trees starts with global_trees, then adds new trees).
    Returns number of trees appended.
    """
    gbt_g = _get_gbtree_section(global_dict)
    gbt_l = _get_gbtree_section(local_dict)

    trees_g: List[Dict[str, Any]] = gbt_g.get("trees", [])
    trees_l: List[Dict[str, Any]] = gbt_l.get("trees", [])

    start = len(trees_g)
    if len(trees_l) <= start:
        return 0

    new_trees = trees_l[start:]
    new_tree_info = (gbt_l.get("tree_info", []) or [0] * len(trees_l))[start:]

    # Append with fresh, consecutive IDs
    next_id = len(trees_g)
    for t in new_trees:
        t = dict(t)  # shallow copy
        _set_tree_id(t, next_id)
        trees_g.append(t)
        next_id += 1

    gbt_g["trees"] = trees_g
    gbt_g["tree_info"] = (gbt_g.get("tree_info", []) or []) + list(new_tree_info)

    # Update params
    num_trees = len(trees_g)
    num_parallel_tree = _get_num_parallel_tree(gbt_g)

    gbt_param = gbt_g.setdefault("gbtree_model_param", {})
    gbt_param["num_trees"] = str(num_trees)
    if "num_parallel_tree" not in gbt_param:
        gbt_param["num_parallel_tree"] = str(num_parallel_tree)

    gbt_g["iteration_indptr"] = _recompute_iteration_indptr(num_trees, num_parallel_tree)
    return len(new_trees)


class XGBoostTreeAppendStrategy(FedAvg):
    """
    FL per XGBoost (orizzontale):
      - Round 1: federated feature selection (come nel tuo progetto)
      - Round >=2: "federated boosting" via *tree appending*
          Ogni client continua l'addestramento partendo dal modello globale
          e il server aggiorna il modello globale appendendo SOLO i nuovi alberi.

    Risultato: UN SOLO modello globale XGBoost (niente ensemble esterno).
    """

    def __init__(
        self,
        top_k: int = 20,
        save_path: str = "selected_features.json",
        local_boost_round: int = 1,
        global_model_path: str = "global_model.json",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.top_k = int(top_k)
        self.save_path = save_path
        self.local_boost_round = int(local_boost_round)

        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None

        self._global_model_bytes: Optional[bytes] = None
        self._global_model_dict: Optional[Dict[str, Any]] = None

        # Root progetto: xgboostmodel/
        self._project_root = Path(__file__).resolve().parents[1]
        self._results_dir = self._project_root / "results"
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # File in results/ (prende solo il nome, anche se passi "results/xxx")
        self._global_model_path = self._results_dir / Path(global_model_path).name
        self._save_path_abs = self._results_dir / Path(save_path).name

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        # Start with empty parameters: round 1 is feature selection anyway.
        if self._global_model_bytes:
            return _params_from_bytes(self._global_model_bytes)
        return ndarrays_to_parameters([])

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)
        new_instructions = []

        if server_round == 1:
            for client_proxy, fit_ins in fit_instructions:
                fit_ins.config["phase"] = "fs"
                fit_ins.config["top_k"] = str(self.top_k)
                fit_ins.config["server_round"] = str(server_round)
                new_instructions.append((client_proxy, fit_ins))
            return new_instructions

        # Round >=2: training (server sends global model params via `parameters`)
        for client_proxy, fit_ins in fit_instructions:
            fit_ins.config["phase"] = "train"
            fit_ins.config["local_boost_round"] = str(self.local_boost_round)
            if self.selected_features is not None:
                fit_ins.config["selected_features"] = json.dumps(self.selected_features)
            fit_ins.config["server_round"] = str(server_round)
            new_instructions.append((client_proxy, fit_ins))
        return new_instructions

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        eval_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        new_instructions = []
        for client_proxy, eval_ins in eval_instructions:
            if self.selected_features is not None:
                eval_ins.config["selected_features"] = json.dumps(self.selected_features)
            eval_ins.config["server_round"] = str(server_round)
            new_instructions.append((client_proxy, eval_ins))
        return new_instructions

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

            # Salvataggi (feature list) -> in xgboostmodel/results/
            save_path_abs = self._save_path_abs
            with open(save_path_abs, "w", encoding="utf-8") as f:
                json.dump({"selected_features": selected}, f, ensure_ascii=False, indent=2)

            global_feat_path = self._results_dir / "global_model_features.json"
            with open(global_feat_path, "w", encoding="utf-8") as f:
                json.dump({"features": list(self.selected_features)}, f, ensure_ascii=False, indent=2)

            print(f"[SERVER] Feature salvate: {global_feat_path}")

            metrics_aggregated: Dict[str, Scalar] = {
                "fs_done": 1.0,
                "n_selected_features": float(len(selected)),
            }
            return ndarrays_to_parameters([]), metrics_aggregated

        # =========================
        # ROUND >=2: TREE APPENDING
        # =========================
        total_examples = 0
        total_train_mae = 0.0
        trees_appended = 0

        # Ensure we have a global dict
        if self._global_model_dict is None and self._global_model_bytes:
            self._global_model_dict = json.loads(self._global_model_bytes.decode("utf-8"))

        for _, fit_res in results:
            total_examples += fit_res.num_examples
            if fit_res.metrics and "train_mae" in fit_res.metrics:
                total_train_mae += float(fit_res.metrics["train_mae"]) * fit_res.num_examples

            # Decode local model bytes
            local_bytes = _bytes_from_params(fit_res.parameters)
            if not local_bytes:
                continue

            try:
                local_dict = json.loads(local_bytes.decode("utf-8"))
            except Exception:
                print("⚠️ Modello client non in formato JSON (serve save_raw(raw_format='json')).")
                continue

            if self._global_model_dict is None:
                # First global model: take the first client model as base
                self._global_model_dict = local_dict
                # Normalize iteration_indptr just in case
                gbt = _get_gbtree_section(self._global_model_dict)
                ntrees = len(gbt.get("trees", []))
                npt = _get_num_parallel_tree(gbt)
                gbt.setdefault("gbtree_model_param", {})["num_trees"] = str(ntrees)
                gbt["iteration_indptr"] = _recompute_iteration_indptr(ntrees, npt)
                continue

            trees_appended += _append_new_trees(self._global_model_dict, local_dict)

        if self._global_model_dict is None:
            print("⚠️ Nessun modello valido ricevuto (round >=2).")
            return _params_from_bytes(self._global_model_bytes), {"trees_appended": 0.0}

        # Persist global model bytes
        self._global_model_bytes = json.dumps(self._global_model_dict).encode("utf-8")
        with open(self._global_model_path, "wb") as f:
            f.write(self._global_model_bytes)

        avg_mae = total_train_mae / total_examples if total_examples > 0 else float("nan")

        # Stats
        gbt = _get_gbtree_section(self._global_model_dict)
        ntrees_global = len(gbt.get("trees", []))

        metrics_aggregated: Dict[str, Scalar] = {
            "train_mae": float(avg_mae) if avg_mae == avg_mae else float("nan"),
            "trees_appended": float(trees_appended),
            "n_trees_global": float(ntrees_global),
        }
        if self.selected_features is not None:
            metrics_aggregated["n_selected_features"] = float(len(self.selected_features))

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
