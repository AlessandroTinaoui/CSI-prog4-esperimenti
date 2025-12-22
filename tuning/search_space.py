# tuning/search_space.py

from __future__ import annotations

from typing import Dict, Any


def suggest_params(trial) -> Dict[str, Any]:
    """
    Ritorna un dict JSON-serializzabile con due sezioni:
      - server: parametri letti da config.py
      - client: parametri letti da model_params.py
    """
    server = {
        "NUM_ROUNDS": trial.suggest_int("server.NUM_ROUNDS", 5, 25),
        "TOP_K_FEATURES": trial.suggest_int("server.TOP_K_FEATURES", 10, 80),
        "LOCAL_BOOST_ROUND": trial.suggest_int("server.LOCAL_BOOST_ROUND", 1, 5),
        # HOLDOUT_CID non lo scegli qui: lo imposta il runner per ogni holdout
    }

    client = {
        "MAX_DEPTH": trial.suggest_int("client.MAX_DEPTH", 2, 10),
        "LEARNING_RATE": trial.suggest_float("client.LEARNING_RATE", 1e-3, 3e-1, log=True),
        "SUBSAMPLE": trial.suggest_float("client.SUBSAMPLE", 0.5, 1.0),
        "COLSAMPLE_BYTREE": trial.suggest_float("client.COLSAMPLE_BYTREE", 0.5, 1.0),
        "REG_LAMBDA": trial.suggest_float("client.REG_LAMBDA", 1e-3, 10.0, log=True),
        "MAX_LOCAL_ROUNDS": trial.suggest_int("client.MAX_LOCAL_ROUNDS", 10, 200),
        "ES_ROUNDS": trial.suggest_int("client.ES_ROUNDS", 3, 30),
        # Opzionale, se vuoi anche la FS con XGBRegressor:
        "N_ESTIMATORS": trial.suggest_int("client.N_ESTIMATORS", 50, 400),
    }

    return {"server": server, "client": client}
