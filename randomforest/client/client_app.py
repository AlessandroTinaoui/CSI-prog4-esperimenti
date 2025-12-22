import os
import sys
import json
import pickle
import logging

import numpy as np
import pandas as pd
import flwr as fl

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from flwr.client import NumPyClient
from flwr.common import FitIns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
from dataset.dataset_cfg import get_train_path

TRAIN_PATH = get_train_path()

from rf_params import (
    RF_INIT_PARAMS,
    RF_FS_PARAMS,
    RF_TRAIN_BASE_PARAMS,
    TREES_PER_ROUND,
)


class RandomForestClient(NumPyClient):
    def __init__(self, cid: int, data_path: str):
        self.cid = cid

        # Load data
        self.data = pd.read_csv(data_path, sep=",")
        log_dir = "../clients_log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"client_{self.cid}_log.txt")

        # Configure logging
        self.logger = logging.getLogger(f"client_{self.cid}")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path)
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
            test_size=0.01,
            random_state=42,
            shuffle=True,
        )

        # Modello RF (inizializzazione base)
        self.model = RandomForestRegressor(**RF_INIT_PARAMS)


    # -------------------------
    # Helper: allena e calcola importanza feature (Round 1)
    # -------------------------
    def _train_and_compute_importance(self, X_train_df: pd.DataFrame, y_train: pd.Series):
        """
        Calcola feature_importances_ (MDI) senza split extra.
        """
        fs_model = RandomForestRegressor(**RF_FS_PARAMS)

        fs_model.fit(X_train_df.values, y_train.values)

        importances = fs_model.feature_importances_
        oob_score = float(getattr(fs_model, "oob_score_", np.nan))
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

        # Allena modello
        # Allena modello (warm-start: parte dal globale e aggiunge alberi)
        self.logger.info("Training the model (Round >=2) - warm start...")

        # 1) deserializza il modello globale se presente
        global_model = None
        if parameters and len(parameters) > 0 and parameters[0] is not None:
            try:
                model_bytes = np.array(parameters[0], dtype=np.uint8).tobytes()
                global_model = pickle.loads(model_bytes)
            except Exception:
                global_model = None

        # 2) usa il modello globale come base, altrimenti crea un RF “vuoto”
        if isinstance(global_model, RandomForestRegressor):
            self.model = global_model
        else:
            self.model = RandomForestRegressor(
                n_estimators=0,
                random_state=42 + self.cid,
                n_jobs=1,
            )

        self.model.set_params(**RF_TRAIN_BASE_PARAMS)

        # 4) aggiunge alberi a ogni round (questo è il punto chiave)
        if hasattr(self.model, "feature_names_in_"):
            X_train = X_train.reindex(columns=list(self.model.feature_names_in_))

        self.model.n_estimators += TREES_PER_ROUND
        self.model.fit(X_train, self.y_train)

        # Training MAE
        y_pred_train = self.model.predict(X_train)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        self.logger.info(f"Model trained. Training MAE: {mae_train:.4f}")

        # Serializza modello (come nel tuo originale)
        model_bytes = pickle.dumps(self.model)
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)

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

        # Deserializza modello
        if parameters and len(parameters) > 0:
            model_bytes = np.array(parameters[0], dtype=np.uint8).tobytes()
            model = pickle.loads(model_bytes)

            if isinstance(model, RandomForestRegressor):
                self.model = model
            else:
                self.logger.info("Deserialized model is not a RandomForestRegressor (uso self.model corrente).")
        else:
            self.logger.info("Nessun modello ricevuto in evaluate, uso self.model corrente.")

        # allinea X_test alle feature con cui il modello è stato fit-tato
        if hasattr(self.model, "feature_names_in_"):
            X_test = X_test.reindex(columns=list(self.model.feature_names_in_))

        # Evaluate MAE
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
    data_path = os.path.join( f"{BASE_DIR}/../{TRAIN_PATH}/group{client_id}_merged_clean.csv")

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    client = RandomForestClient(cid=client_id, data_path=data_path)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
