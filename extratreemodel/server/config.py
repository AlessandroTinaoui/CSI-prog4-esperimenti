# Configurazione per il server Flower (ExtraTrees)
# default
# server/config.py
import os

SERVER_ADDRESS = "127.0.0.1:8080"

if "FL_SERVER_ADDRESS" in os.environ:
    SERVER_ADDRESS = os.environ["FL_SERVER_ADDRESS"]

HOLDOUT_CID = int(os.environ.get("HOLDOUT_CID", 0))

# Numero di round Flower:
# round 1 = feature selection
# round 2..NUM_ROUNDS = training + build ensemble
NUM_ROUNDS = 2

# Client tenuto fuori dal training e usato come holdout finale (0..8). Metti 9 o 10 per disattivare.
HOLDOUT_CID = 7
# Feature selection
TOP_K_FEATURES = 32

# Quanti modelli tenere nell'ensemble globale (se None -> tutti quelli ricevuti)
MAX_MODELS_IN_ENSEMBLE = None

