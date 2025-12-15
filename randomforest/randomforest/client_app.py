import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from flwr.client import NumPyClient
from flwr.common import FitIns, FitRes
import sys
import pickle
import logging
import flwr as fl
import numpy as np
import os

class RandomForestClient(NumPyClient):
    def __init__(self, cid: int, data_path: str):
        self.cid = cid
        # Load data
        self.data = pd.read_csv(data_path, sep=',')
        
        # Configure logging
        self.logger = logging.getLogger(f"client_{self.cid}")
        handler = logging.FileHandler(f"client_{self.cid}_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Log dataset info
        self.logger.info(f"Columns in dataset: {self.data.columns}")
        self.logger.info(f"First few rows:\n{self.data.head()}")
        
        self.data = self.data.dropna()

        # Drop unnecessary columns
        cols_to_drop = ["day", "client_id", "user_id", "source_file"]
        # Drop only existing columns to avoid errors if some are missing
        cols_to_drop = [c for c in cols_to_drop if c in self.data.columns]
        
        self.X = self.data.drop(columns=cols_to_drop + ["label"])
        self.y = self.data["label"]

        # Train/Test split (80/20)
        split_idx = int(0.8 * len(self.X))
        self.X_train = self.X.iloc[:split_idx]
        self.y_train = self.y.iloc[:split_idx]
        self.X_test = self.X.iloc[split_idx:]
        self.y_test = self.y.iloc[split_idx:]

        # Initialize RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

    def fit(self, parameters, ins: FitIns):
        """Train the Random Forest model."""
        self.logger.info("Training the model...")
        self.model.fit(self.X_train, self.y_train)

        # Calculate training MAE
        y_pred_train = self.model.predict(self.X_train)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        self.logger.info(f"Model trained. Training MAE: {mae_train:.4f}")
        
        # Serialize model
        model_bytes = pickle.dumps(self.model)
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)

        return [model_array], len(self.X_train), {"train_mae": mae_train}

    def evaluate(self, parameters, ins: FitIns):
        """Evaluate the model."""
        model_bytes = np.array(parameters[0], dtype=np.uint8).tobytes()
    
        # Deserialize the model
        model = pickle.loads(model_bytes)
        
        if isinstance(model, RandomForestRegressor):
            print("Successfully deserialized RandomForestRegressor model.")
            self.model = model
        else:
            print("Deserialized model is not a RandomForestRegressor.")
            
        # Calculate evaluation MAE
        y_pred_test = self.model.predict(self.X_test)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        
        print(f"Client {self.cid} - Evaluation MAE: {mae_test:.4f}")
        self.logger.info(f"Model evaluation: MAE = {mae_test:.4f}")
        
        return float(mae_test), len(self.X_test), {"eval_mae": mae_test}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_app.py <client_id>")
        sys.exit(1)
        
    client_id = int(sys.argv[1])
    # Update path to match the new file structure: group{id}_merged_clean.csv
    data_path = f"clients_data/group{client_id}_merged_clean.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        sys.exit(1)
        
    client = RandomForestClient(cid=client_id, data_path=data_path)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
