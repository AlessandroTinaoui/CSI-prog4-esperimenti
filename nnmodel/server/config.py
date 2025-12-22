# nnmodel/server/config.py

SERVER_ADDRESS = "127.0.0.1:8080"

# Round 1: calcolo scaler globale federato
# Round 2..NUM_ROUNDS: training FedAvg
NUM_ROUNDS = 1000

# Holdout client (0..8) da tenere fuori dal training (come fai già) :contentReference[oaicite:4]{index=4}
HOLDOUT_CID = 2  # metti 9 o 10 per disattivare

# Federated settings
FRACTION_FIT = 1.0
FRACTION_EVALUATE = 1.0
MIN_FIT_CLIENTS = 8
MIN_EVALUATE_CLIENTS = 8
MIN_AVAILABLE_CLIENTS = 8

# Training hyperparams (client-side)
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Model
HIDDEN_SIZES = [64, 32, 16]
DROPOUT = 0.2

# Target range (Garmin sleep score tipico 0..100)
CLIP_MIN = 0.0
CLIP_MAX = 100.0

# Split
TEST_SIZE = 0.2
RANDOM_STATE = 42
SHUFFLE_SPLIT = True

# Dove salvare risultati globali (feature list + scaler + modello)
RESULTS_DIRNAME = "results"
GLOBAL_FEATURES_JSON = "global_features.json"
GLOBAL_SCALER_JSON = "global_scaler.json"
GLOBAL_MODEL_PTH = "global_model.pth"
