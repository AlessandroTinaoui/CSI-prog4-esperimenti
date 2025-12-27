# rf_params.py
# =========================
# Parametri Random Forest – Federated Learning
# =========================

# -------------------------
# RandomForest iniziale (costruttore base)
# -------------------------
RF_INIT_PARAMS = {
    "n_estimators": 0,          # parte vuoto (warm start)
    "random_state": 42,
    "n_jobs": 1,
}

# -------------------------
# RandomForest per FEATURE SELECTION (Round 1)
# -------------------------
RF_FS_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": 1,
    "bootstrap": True,
    "oob_score": True,
    "min_samples_leaf": 5,
}

# -------------------------
# Parametri di TRAINING (Round >=2)
# -------------------------
RF_TRAIN_BASE_PARAMS = {
    "warm_start": True,
    "bootstrap": True,
    "max_samples": 0.7,  # diversifica gli alberi (bagging più “aggressivo”)
    "max_features": 0.6,  # più randomizzazione => più varianza
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_depth": 15,  # limite “ragionevole” (paper usa 15 in setup di base)
    "n_jobs": 1,
    "random_state": 42,  # oppure 42+client_id per più diversità tra client
}

TREES_PER_ROUND = 10
# -------------------------
# Federated settings
# -------------------------
#TREES_PER_ROUND = 50          # alberi aggiunti a ogni round
