# Configurazione per il server Flower

SERVER_ADDRESS = "localhost:8080"

# Numero di round Flower (nota: round 1 = feature selection, round 2..NUM_ROUNDS = training)
NUM_ROUNDS = 2

# Client tenuto fuori dal training e usato come holdout finale (0..8). Metti 9 o 10 per disattivare.
HOLDOUT_CID = 2

# Feature selection
TOP_K_FEATURES = 30

# Quanti alberi XGBoost aggiunge ogni client ad ogni round (round >= 2).
# Con 9 client e non-IID, valori piccoli (1-2) sono spesso più stabili.
LOCAL_BOOST_ROUND = 1
