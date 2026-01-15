# client_params.py

# --- DATA PIPELINE ---
CLIP_MIN = 0.0
CLIP_MAX = 100.0

TEST_SIZE = 0.2
RANDOM_STATE = 42
SHUFFLE_SPLIT = True

# --- TRAINING ---
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5

# FedProx: forza il modello locale a non allontanarsi troppo dal globale.
FEDPROX_MU = 0.01

# --- TABNET ARCH ---
TABNET_N_D = 24
TABNET_N_A = 24
TABNET_N_STEPS = 5
TABNET_GAMMA = 1.5
TABNET_N_SHARED = 2
TABNET_N_INDEPENDENT = 2
TABNET_BN_VIRTUAL_BS = 128
TABNET_BN_MOMENTUM = 0.02

# --- LOSS ---
# "mae" o "huber"
LOSS_NAME = "mae"
HUBER_BETA = 2.0
