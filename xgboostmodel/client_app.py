import os
import sys
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import flwr as fl
import xgboost as xgb
from flwr.client import NumPyClient
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def _bytes_from_ndarrays(parameters: List[np.ndarray]) -> Optional[bytes]:
    if not parameters:
        return None
    arr = parameters[0]
    if arr.size == 0:
        return None
    return np.array(arr, dtype=np.uint8).tobytes()


def _load_booster_from_json_bytes(model_bytes: bytes) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(bytearray(model_bytes))
    return bst


class XGBoostClient(NumPyClient):
    """
    - Round 1 (phase="fs"): invia feature importances (no model)
    - Round >=2 (phase="train"): riceve modello globale (JSON bytes) + selected_features,
      continua l'addestramento per `local_boost_round` e invia il modello aggiornato (global+new trees).
    """

    def __init__(self, cid: int, data_path: str):
        self.cid = int(cid)
        self.data = pd.read_csv(data_path, sep=",").dropna()

        self.logger = logging.getLogger(f"client_{self.cid}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(f"client_{self.cid}_log.txt")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        cols_to_drop = ["day", "client_id", "user_id", "source_file"]
        self.cols_to_drop = [c for c in cols_to_drop if c in self.data.columns]

        self.target_col = "label"
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' non trovata nel dataset")

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

    def _train_and_compute_importance(self, X_train_df: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42 + self.cid,
            n_jobs=1,
            objective="reg:squarederror",
            verbosity=0,
        )
        model.fit(X_train_df, y_train)
        return np.array(model.feature_importances_, dtype=float)

    def fit(self, parameters, config):
        phase = (config or {}).get("phase", "train")

        if phase == "fs":
            self.logger.info("Round 1: FEATURE SELECTION.")
            importances = self._train_and_compute_importance(self.X_train_full, self.y_train)
            metrics = {
                "local_feature_importance": json.dumps(importances.tolist()),
                "feature_names": json.dumps(self.feature_names_full),
            }
            return [], len(self.X_train_full), metrics

        selected = (config or {}).get("selected_features", None)
        if selected is None:
            selected_features = list(self.feature_names_full)
        else:
            selected_features = json.loads(selected) if isinstance(selected, str) else list(selected)
            selected_set = set(selected_features)
            selected_features = [f for f in self.feature_names_full if f in selected_set]

        self.selected_features = list(selected_features)
        local_boost_round = int((config or {}).get("local_boost_round", 1))

        X_train = self.X_train_full[self.selected_features]
        X_test = self.X_test_full[self.selected_features]

        dtrain = xgb.DMatrix(X_train, label=self.y_train, feature_names=self.selected_features)
        dtest = xgb.DMatrix(X_test, label=self.y_test, feature_names=self.selected_features)

        global_bytes = _bytes_from_ndarrays(parameters)
        booster_in: Optional[xgb.Booster] = None
        if global_bytes:
            try:
                booster_in = _load_booster_from_json_bytes(global_bytes)
            except Exception as e:
                self.logger.info(f"Impossibile caricare modello globale: {e}")
                booster_in = None

        xgb_params: Dict[str, object] = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
            "seed": 42 + self.cid,
            "verbosity": 0,
        }

        bst = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=local_boost_round,
            xgb_model=booster_in,
            evals=[(dtrain, "train"), (dtest, "test")],
            verbose_eval=False,
        )

        y_pred_train = bst.predict(dtrain)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)

        # Server vuole JSON per poter fare append dei tree
        raw_json = bst.save_raw(raw_format="json")
        model_array = np.frombuffer(raw_json, dtype=np.uint8)

        metrics = {
            "train_mae": float(mae_train),
            "n_features": int(len(self.selected_features)),
            "local_boost_round": int(local_boost_round),
        }
        return [model_array], len(X_train), metrics

    def evaluate(self, parameters, config):
        selected = (config or {}).get("selected_features", None)
        if selected is None:
            selected_features = self.selected_features or list(self.feature_names_full)
        else:
            selected_features = json.loads(selected) if isinstance(selected, str) else list(selected)
            selected_set = set(selected_features)
            selected_features = [f for f in self.feature_names_full if f in selected_set]

        X_test = self.X_test_full[selected_features]
        dtest = xgb.DMatrix(X_test, label=self.y_test, feature_names=selected_features)

        model_bytes = _bytes_from_ndarrays(parameters)
        if not model_bytes:
            return float("nan"), len(X_test), {"eval_mae": float("nan")}

        try:
            bst = _load_booster_from_json_bytes(model_bytes)
            y_pred = bst.predict(dtest)
            mae = mean_absolute_error(self.y_test, y_pred)
            return float(mae), len(X_test), {"eval_mae": float(mae)}
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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, "clients_data", f"group{client_id}_merged_clean.csv")

    if not os.path.exists(data_path):
        print(f"ERRORE: Data file not found: {data_path}")
        sys.exit(1)

    client = XGBoostClient(cid=client_id, data_path=data_path)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
