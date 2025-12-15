# client_app.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flwr.client import NumPyClient
from flwr.common import FitIns
import sys
import pickle
import logging
import flwr as fl
import numpy as np
import json

from server.feature_selection_utils import compute_l1_stats, compute_l2_permutation_importance

TARGET_COL = "label"      # <-- cambia in "sleep_score" se quello è il tuo target
DROP_COLS = ["date"]      # <-- metti qui colonne da ignorare sempre (id, date, ecc.)
TASK_TYPE = "regression"  # <-- "regression" se sleep_score è continuo


class RandomForestClient(NumPyClient):
    def __init__(self, cid: int, data_path: str):
        self.cid = cid
        self.data = pd.read_csv(data_path, sep=';')

        # logging come nel tuo codice
        self.logger = logging.getLogger(f"client_{self.cid}")
        handler = logging.FileHandler(f"client_{self.cid}_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.logger.info(f"Columns in dataset: {self.data.columns}")
        self.logger.info(f"First few rows:\n{self.data.head()}")

        # NON fare dropna qui: serve per missing-rate nel livello 1
        # self.data = self.data.dropna()

        # Modello (lo manteniamo come nel tuo esempio)
        self.model = RandomForestClassifier(n_estimators=50, max_depth=10)

    def fit(self, parameters, ins: FitIns):
        task = ins.config.get("task", "train")

        # -------- ROUND 1: feature selection (L1 stats + L2 permutation importance)
        if task == "feature_select":
            l1 = compute_l1_stats(self.data, target_col=TARGET_COL, drop_cols=DROP_COLS)
            l2 = compute_l2_permutation_importance(
                self.data,
                target_col=TARGET_COL,
                drop_cols=DROP_COLS,
                task_type=TASK_TYPE,
                n_estimators=200,
                n_repeats=5,
                test_size=0.2,
                random_state=42,
            )

            metrics = {
                "l1_n": l1.n,
                "l1_missing_json": json.dumps(l1.missing_rate),
                "l1_var_json": json.dumps(l1.variance),
                "l1_features_json": json.dumps(l1.feature_names),
                "l2_importances_json": json.dumps(l2),
            }

            self.logger.info("Sent L1/L2 stats for feature selection.")
            return parameters, 0, metrics

        # -------- ROUND 2+: training normale con selected_features
        selected_features = json.loads(ins.config.get("selected_features_json", "[]"))

        # prepara X,y
        X = self.data.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
        X = X.select_dtypes(include=[np.number]).copy()

        if selected_features:
            keep = [c for c in selected_features if c in X.columns]
            X = X[keep].copy()

        y = self.data[TARGET_COL]

        # imputazione semplice
        X = X.fillna(X.median(numeric_only=True))

        # split semplice (NON time-series come richiesto)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.logger.info(f"Training with {X.shape[1]} selected features.")

        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_train, y_train)
        self.logger.info(f"Model trained. Accuracy: {accuracy:.4f}")

        model_bytes = pickle.dumps(self.model)
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        return [model_array], len(X_train), {"train_accuracy": float(accuracy)}

    def evaluate(self, parameters, ins: FitIns):
        model_bytes = np.array(parameters[0], dtype=np.uint8).tobytes()
        model = pickle.loads(model_bytes)
        self.model = model

        # stesso schema dati di fit (con selected_features)
        import json
        selected_features = json.loads(ins.config.get("selected_features_json", "[]"))

        X = self.data.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
        X = X.select_dtypes(include=[np.number]).copy()
        if selected_features:
            keep = [c for c in selected_features if c in X.columns]
            X = X[keep].copy()

        y = self.data[TARGET_COL]
        X = X.fillna(X.median(numeric_only=True))

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        accuracy = self.model.score(X_test, y_test)
        self.logger.info(f"Model evaluation: Accuracy = {accuracy:.4f}")
        return 0.0, len(X_test), {"eval_accuracy": float(accuracy)}


if __name__ == "__main__":
    client_id = int(sys.argv[1])
    data_path = f"clients_data/group{client_id}_merged_clean.csv"
    client = RandomForestClient(cid=client_id, data_path=data_path)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
