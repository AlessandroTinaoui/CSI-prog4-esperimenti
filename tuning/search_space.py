# tuning/search_space.py
from __future__ import annotations

from typing import Any, Dict


def _server_space(trial) -> Dict[str, Any]:
    # Parametri lato server (comuni a tutti i modelli)
    return {
        "NUM_ROUNDS": trial.suggest_int("server.NUM_ROUNDS", 30, 200),
        "TOP_K_FEATURES": trial.suggest_int("server.TOP_K_FEATURES", 10, 50),
        #"LOCAL_BOOST_ROUND": trial.suggest_int("server.LOCAL_BOOST_ROUND", 1, 3),
        # HOLDOUT_CID lo imposta il runner holdout (non qui)
    }


def _xgboost_space(trial) -> Dict[str, Any]:
    return {
        "NUM_ROUNDS": trial.suggest_int("server.NUM_ROUNDS", 30, 400),
        "TOP_K_FEATURES": trial.suggest_int("server.TOP_K_FEATURES", 10, 60),

        "N_BINS": trial.suggest_categorical("server.N_BINS", [16, 32, 64, 128]),
        "HUBER_DELTA": trial.suggest_float("server.HUBER_DELTA", 0.05, 5.0, log=True),
        "REG_LAMBDA": trial.suggest_float("server.REG_LAMBDA", 1e-3, 50.0, log=True),
        "GAMMA": trial.suggest_float("server.GAMMA", 0.0, 5.0),
        "LEARNING_RATE": trial.suggest_float("server.LEARNING_RATE", 1e-3, 0.5, log=True),

        # opzionale: spesso meglio fissarlo o calcolarlo, vedi sotto
        "BASE_SCORE": trial.suggest_float("server.BASE_SCORE", -5.0, 5.0),
    }


def _randomforest_space(trial) -> Dict[str, Any]:
    """
    Spazio tipico per RandomForestRegressor.
    Chiavi proposte (tutte sotto "client"):
      - N_ESTIMATORS
      - MAX_DEPTH (può essere None: vedi MAX_DEPTH_IS_NONE)
      - MAX_FEATURES
      - MIN_SAMPLES_SPLIT
      - MIN_SAMPLES_LEAF
      - BOOTSTRAP
      - CRITERION
    """
    max_depth_is_none = trial.suggest_categorical("client.MAX_DEPTH_IS_NONE", [False, True])
    max_depth_val = None if max_depth_is_none else trial.suggest_int("client.MAX_DEPTH", 2, 40)

    return {
        "MODEL_TYPE": "randomforest",  # utile se vuoi distinguere nel client
        "N_ESTIMATORS": trial.suggest_int("client.N_ESTIMATORS", 50, 800),

        # se MAX_DEPTH_IS_NONE=True, MAX_DEPTH verrà messo a None
        "MAX_DEPTH": max_depth_val,
        "MAX_DEPTH_IS_NONE": max_depth_is_none,

        # max_features: in sklearn può essere float (0-1), int, "sqrt", "log2"
        "MAX_FEATURES": trial.suggest_categorical(
            "client.MAX_FEATURES",
            ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
        ),

        "MIN_SAMPLES_SPLIT": trial.suggest_int("client.MIN_SAMPLES_SPLIT", 2, 20),
        "MIN_SAMPLES_LEAF": trial.suggest_int("client.MIN_SAMPLES_LEAF", 1, 20),

        "BOOTSTRAP": trial.suggest_categorical("client.BOOTSTRAP", [True, False]),

        # criterio dipende dalla tua implementazione; in sklearn: "squared_error" o "absolute_error"
        "CRITERION": trial.suggest_categorical(
            "client.CRITERION",
            ["squared_error", "absolute_error"],
        ),
    }


def _extratree_space(trial) -> Dict[str, Any]:
    """
    Spazio tipico per ExtraTreesRegressor.
    Chiavi proposte (tutte sotto "client"):
      - N_ESTIMATORS
      - MAX_DEPTH (None gestito come per RF)
      - MAX_FEATURES
      - MIN_SAMPLES_SPLIT
      - MIN_SAMPLES_LEAF
      - BOOTSTRAP (per ExtraTrees spesso False, ma lo lasciamo ottimizzabile)
      - CRITERION
    """
    max_depth_is_none = trial.suggest_categorical("client.MAX_DEPTH_IS_NONE", [False, True])
    max_depth_val = None if max_depth_is_none else trial.suggest_int("client.MAX_DEPTH", 2, 60)

    return {
        "MODEL_TYPE": "extratree",
        "N_ESTIMATORS": trial.suggest_int("client.N_ESTIMATORS", 50, 1200),

        "MAX_DEPTH": max_depth_val,
        "MAX_DEPTH_IS_NONE": max_depth_is_none,

        "MAX_FEATURES": trial.suggest_categorical(
            "client.MAX_FEATURES",
            ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
        ),

        "MIN_SAMPLES_SPLIT": trial.suggest_int("client.MIN_SAMPLES_SPLIT", 2, 30),
        "MIN_SAMPLES_LEAF": trial.suggest_int("client.MIN_SAMPLES_LEAF", 1, 30),

        # in ExtraTrees spesso bootstrap=False; lo rendiamo ottimizzabile
        "BOOTSTRAP": trial.suggest_categorical("client.BOOTSTRAP", [False, True]),

        "CRITERION": trial.suggest_categorical(
            "client.CRITERION",
            ["squared_error", "absolute_error"],
        ),
    }
def suggest_nn_params(trial):
    # server
    # CLIENT (training + architettura)
    client = {
        "LOCAL_EPOCHS": trial.suggest_int("client.LOCAL_EPOCHS", 1, 5),
        "BATCH_SIZE": trial.suggest_categorical("client.BATCH_SIZE", [16, 32, 64]),
        "LR": trial.suggest_float("client.LR", 1e-4, 5e-3, log=True),
        "WEIGHT_DECAY": trial.suggest_float("client.WEIGHT_DECAY", 1e-6, 5e-3, log=True),
        "DROPOUT": trial.suggest_float("client.DROPOUT", 0.0, 0.5),

        # 3 layer come default, ma dimensioni ottimizzate
       # "HIDDEN_SIZES": [
        #    trial.suggest_int("client.H1", 16, 256, log=True),
         #   trial.suggest_int("client.H2", 16, 128, log=True),
          #  trial.suggest_int("client.H3", 8, 64, log=True),
        #],
    }


    return  client


def suggest_params(trial) -> Dict[str, Any]:
    """
    Ritorna un dict JSON-serializzabile con:
      - "server": parametri server (comuni)
      - "client": parametri del modello selezionato

    IMPORTANTISSIMO:
      Lo studio driver deve fare: trial.set_user_attr("model", args.model)
      così qui sappiamo quale spazio usare.
    """
    model = trial.user_attrs.get("model", "xgboostmodel")

    server = _server_space(trial)

    if model == "xgboostmodel":
        server = _xgboost_space(trial)
        client = {}
    elif model == "randomforest":
        client = _randomforest_space(trial)
    elif model == "extratree":
        client = _extratree_space(trial)
    elif model == "mlp":
        client = suggest_nn_params(trial)
    else:
        raise ValueError(f"Modello non supportato nello search space: {model}")

    return {"server": server, "client": client}
