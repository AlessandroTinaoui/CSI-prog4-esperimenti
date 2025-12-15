import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flwr.client import NumPyClient
from flwr.common import FitIns, FitRes
import sys
import pickle
import logging
import flwr as fl
import numpy as np

class RandomForestClient(NumPyClient):
    def __init__(self, cid: int, data_path: str):
        self.cid = cid
        self.data = pd.read_csv(data_path, sep=';')  # Usa il separatore corretto

        # Configurazione del logging per ogni client
        self.logger = logging.getLogger(f"client_{self.cid}")
        handler = logging.FileHandler(f"client_{self.cid}_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Log informazioni sul dataset
        self.logger.info(f"Columns in dataset: {self.data.columns}")
        self.logger.info(f"First few rows:\n{self.data.head()}")
        
        self.data = self.data.dropna()

        self.X = self.data.drop(columns=["date", "label"]) 
        self.y = self.data["label"]  

        self.X_train = self.X.iloc[:int(0.8 * len(self.X))]
        self.y_train = self.y.iloc[:int(0.8 * len(self.y))]
        self.X_test = self.X.iloc[int(0.8 * len(self.X)):]
        self.y_test = self.y.iloc[int(0.8 * len(self.y)):]

        self.model = RandomForestClassifier(n_estimators=50, max_depth=10)

    def fit(self, parameters, ins: FitIns):
        print(parameters)
        """Allenamento del modello Random Forest"""
        self.logger.info("Training the model...")
        self.model.fit(self.X, self.y)

        accuracy = self.model.score(self.X_train, self.y_train)
        self.logger.info(f"Model trained. Accuracy: {accuracy:.4f}")
        
        # Serializza il modello per inviarlo al server
        model_bytes = pickle.dumps(self.model)
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)

        return [model_array], len(self.X_train), {"train_accuracy": accuracy}  # Restituisci il modello allenato

    def evaluate(self, parameters, ins: FitIns):
        """Valutazione del modello"""
        model_bytes = np.array(parameters[0], dtype=np.uint8).tobytes()
    
        # Deserialize the model from the byte array
        model = pickle.loads(model_bytes)
        
        if isinstance(model, RandomForestClassifier):
            print("Successfully deserialized RandomForest model.")
            self.model = model
            print(self.model)
        else:
            print("Deserialized model is not a RandomForest.")
            
        accuracy = self.model.score(self.X_test, self.y_test)
        print(accuracy)
        self.logger.info(f"Model evaluation: Accuracy = {accuracy:.4f}")
        return 0., len(self.X_test), {"eval_accuracy": accuracy}

if __name__ == "__main__":
    client_id = int(sys.argv[1])  
    data_path = f"clients_data/client_{client_id}.csv" 
    client = RandomForestClient(cid=client_id, data_path=data_path)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
