import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import flwr as fl
import xgboost as xgb

from sklearn.metrics import mean_absolute_error
from flwr.client import NumPyClient
from flwr.common import FitIns
from xgboost import XGBRegressor


class XGBoostClient(NumPyClient):
    def __init__(self, cid: int, data_path: str):
        self.cid = cid

        # Load data
        print(data_path)
        self.data = pd.read_csv(data_path, sep=",")

        # Configure logging
        self.logger = logging.getLogger(f"client_{self.cid}")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"client_{self.cid}_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)

        # Evita doppio handler se il client viene istanziato più volte
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        # Log dataset info
        self.logger.info(f"Columns in dataset: {self.data.columns.tolist()}")
        self.logger.info(f"First few rows:\n{self.data.head()}")

        # Pulizia base
        self.data = self.data.dropna()

        # Drop unnecessary columns (solo se esistono)
        self.cols_to_drop = ["day", "client_id", "user_id", "source_file"]
        self.cols_to_drop = [c for c in self.cols_to_drop if c in self.data.columns]

        # Target
        self.target_col = "label"
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' non trovata nel dataset")

        # Feature dataframe completo (prima della feature selection)
        self.X_full = self.data.drop(columns=self.cols_to_drop + [self.target_col])
        self.y_full = self.data[self.target_col]

        # Feature names (ordine stabile!)
        self.feature_names_full = self.X_full.columns.tolist()

        # Split Train/Test (80/20) - come nel tuo originale
        from sklearn.model_selection import train_test_split

        self.X_train_full, self.X_test_full, self.y_train, self.y_test = train_test_split(
            self.X_full,
            self.y_full,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

        # Modello RF (inizializzazione base)
        self.model = XGBRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
        )

    # -------------------------
    # Helper: allena e calcola importanza feature (Round 1)
    # -------------------------
    def _train_and_compute_importance(self, X_train_df: pd.DataFrame, y_train: pd.Series):
        """Allena un piccolo XGBoost e ritorna (importance, oob_score_placeholder)."""
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

        # feature_importances_ (gain-based in XGBoost sklearn wrapper)
        importances = np.array(model.feature_importances_, dtype=float)
        # placeholder (non esiste OOB per XGBoost)
        oob_score = float("nan")
        return importances, oob_score


    # -------------------------
    # FIT
    # -------------------------
    def fit(self, parameters, ins: FitIns):
        """
        Round 1 (phase="fs"):
          - calcola importance + feature_names
          - NON invia modello (il server in Round 1 ignora il modello)
        Round >=2 (phase="train"):
          - riceve selected_features
          - filtra X_train/X_test
          - allena RF
          - serializza modello e lo invia come parameters
        """
        config = ins.config if hasattr(ins, "config") else {}
        phase = config.get("phase", "train")

        # -------------------------
        # ROUND 1: FEATURE SELECTION
        # -------------------------
        if phase == "fs":
            self.logger.info("Round 1: FEATURE SELECTION (calcolo importanze).")

            importances, oob_score = self._train_and_compute_importance(
                self.X_train_full, self.y_train
            )

            metrics = {
                "local_feature_importance": importances.tolist(),
                "feature_names": json.dumps(self.feature_names_full),
                "oob_score": float(oob_score),
            }

            # Non inviamo un modello in parameters (ritorniamo array vuoto)
            # così il server può ignorare parameters in Round 1.
            return [], len(self.X_train_full), metrics

        # -------------------------
        # ROUND >=2: TRAIN (con selected_features)
        # -------------------------
        selected = config.get("selected_features", None)

        if selected is None:
            selected_features = self.feature_names_full
            self.logger.info("Nessuna selected_features ricevuta: uso tutte le feature.")
        else:
            if isinstance(selected, str):
                selected_features = json.loads(selected)
            else:
                selected_features = list(selected)

            self.logger.info(f"Received selected_features (n={len(selected_features)}): {selected_features}")

        # Filtra train/test mantenendo ordine
        X_train = self.X_train_full[selected_features]
        X_test = self.X_test_full[selected_features]

        # -------------------------
        # Allena XGBoost (da zero ogni round)
        # -------------------------
        self.logger.info("Training XGBoost model (Round >=2)...")

        self.model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42 + self.cid,
            n_jobs=1,  # più stabile in ambienti condivisi
            objective="reg:squarederror",
            verbosity=0,
        )

        self.model.fit(X_train, self.y_train)

        # Evaluate train MAE (solo per logging/metriche FL)
        y_pred_train = self.model.predict(X_train)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)

        # Serializza booster come bytes (save_raw) e invia come uint8 ndarray
        raw = self.model.get_booster().save_raw()
        model_array = np.frombuffer(raw, dtype=np.uint8)

        metrics = {
            "train_mae": float(mae_train),
            "n_features": int(len(selected_features)),
        }

        return [model_array], len(X_train), metrics


    # -------------------------
    # EVALUATE
    # -------------------------
    def evaluate(self, parameters, ins: FitIns):
        """
        Valutazione: deserializza modello e calcola MAE sul test set filtrato (se selected_features esiste).
        """
        config = ins.config if hasattr(ins, "config") else {}
        selected = config.get("selected_features", None)

        if selected is None:
            selected_features = self.feature_names_full
        else:
            if isinstance(selected, str):
                selected_features = json.loads(selected)
            else:
                selected_features = list(selected)

        X_test = self.X_test_full[selected_features]
        # Valuto il modello locale allenato nell'ultimo fit (il server non invia un modello globale)
        if self.model is None:
            self.logger.info("Nessun modello locale disponibile in evaluate (fit non ancora eseguito?).")
            return float("nan"), len(X_test), {"eval_mae": float("nan")}

        y_pred_test = self.model.predict(X_test)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)

        print(f"Client {self.cid} - Evaluation MAE: {mae_test:.4f}")
        self.logger.info(f"Model evaluation: MAE = {mae_test:.4f}")

        return float(mae_test), len(X_test), {"eval_mae": float(mae_test)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_app.py <client_id>")
        sys.exit(1)

    client_id = int(sys.argv[1])

    # group{id}_merged_clean.csv come nel tuo originale
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "clients_data", f"group{client_id}_merged_clean.csv")
    print("Reading:", data_path)

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    client = XGBoostClient(cid=client_id, data_path=data_path)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
