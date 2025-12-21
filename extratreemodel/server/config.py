# Configurazione per il server Flower (ExtraTrees)

SERVER_ADDRESS = "127.0.0.1:8080"

# Numero di round Flower:
# round 1 = feature selection
# round 2..NUM_ROUNDS = training + build ensemble
NUM_ROUNDS = 2

# Client tenuto fuori dal training e usato come holdout finale (0..8). Metti 9 o 10 per disattivare.
HOLDOUT_CID = 3

# Feature selection
TOP_K_FEATURES = 30

# ExtraTrees hyperparams (uguali per tutti i client per coerenza)
ET_N_ESTIMATORS = 300
ET_MAX_DEPTH = None
ET_MIN_SAMPLES_SPLIT = 2
ET_MIN_SAMPLES_LEAF = 1
ET_MAX_FEATURES = 1.0  # oppure 1.0 / "sqrt" (dipende dal tuo dataset)
ET_BOOTSTRAP = False

# Quanti modelli tenere nell'ensemble globale (se None -> tutti quelli ricevuti)
MAX_MODELS_IN_ENSEMBLE = None
