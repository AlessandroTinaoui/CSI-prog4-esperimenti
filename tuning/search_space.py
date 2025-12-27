# tuning/search_space.py
from __future__ import annotations

from typing import Any, Dict


def _server_space(trial) -> Dict[str, Any]:
    # Parametri lato server (comuni a tutti i modelli)
    return {
        "NUM_ROUNDS": trial.suggest_int("server.NUM_ROUNDS", 2, 2),
        "TOP_K_FEATURES": trial.suggest_int("server.TOP_K_FEATURES", 10, 50),
        "LOCAL_BOOST_ROUND": trial.suggest_int("server.LOCAL_BOOST_ROUND", 1, 3),
        # HOLDOUT_CID lo imposta il runner holdout (non qui)
    }


def _xgboost_space(trial) -> Dict[str, Any]:
    return {
        "N_ESTIMATORS": trial.suggest_int("client.N_ESTIMATORS", 50, 500),
        "MAX_DEPTH": trial.suggest_int("client.MAX_DEPTH", 2, 12),
        "LEARNING_RATE": trial.suggest_float("client.LEARNING_RATE", 1e-3, 3e-1, log=True),
        "SUBSAMPLE": trial.suggest_float("client.SUBSAMPLE", 0.5, 1.0),
        "COLSAMPLE_BYTREE": trial.suggest_float("client.COLSAMPLE_BYTREE", 0.5, 1.0),
        "REG_LAMBDA": trial.suggest_float("client.REG_LAMBDA", 1e-3, 10.0, log=True),
        "MAX_LOCAL_ROUNDS": trial.suggest_int("client.MAX_LOCAL_ROUNDS", 10, 300),
        "ES_ROUNDS": trial.suggest_int("client.ES_ROUNDS", 3, 50),
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
        client = _xgboost_space(trial)
    elif model == "randomforest":
        client = _randomforest_space(trial)
    elif model == "extratree":
        client = _extratree_space(trial)
    else:
        raise ValueError(f"Modello non supportato nello search space: {model}")

    return {"server": server, "client": client}
