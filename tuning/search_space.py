# tuning/search_space.py
from __future__ import annotations

from typing import Any, Dict


def _server_space(trial) -> Dict[str, Any]:
    return {
        "NUM_ROUNDS": trial.suggest_int("server.NUM_ROUNDS", 30, 200),
        "TOP_K_FEATURES": trial.suggest_int("server.TOP_K_FEATURES", 10, 50),
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
    max_depth_is_none = trial.suggest_categorical("client.MAX_DEPTH_IS_NONE", [False, True])
    max_depth_val = None if max_depth_is_none else trial.suggest_int("client.MAX_DEPTH", 2, 40)

    return {
        "MODEL_TYPE": "randomforest",
        "N_ESTIMATORS": trial.suggest_int("client.N_ESTIMATORS", 50, 800),
        "MAX_DEPTH": max_depth_val,
        "MAX_DEPTH_IS_NONE": max_depth_is_none,
        "MAX_FEATURES": trial.suggest_categorical(
            "client.MAX_FEATURES", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]
        ),
        "MIN_SAMPLES_SPLIT": trial.suggest_int("client.MIN_SAMPLES_SPLIT", 2, 20),
        "MIN_SAMPLES_LEAF": trial.suggest_int("client.MIN_SAMPLES_LEAF", 1, 20),
        "BOOTSTRAP": trial.suggest_categorical("client.BOOTSTRAP", [True, False]),
        "CRITERION": trial.suggest_categorical(
            "client.CRITERION", ["squared_error", "absolute_error"]
        ),
    }


def _extratree_space(trial) -> Dict[str, Any]:
    max_depth_is_none = trial.suggest_categorical("client.MAX_DEPTH_IS_NONE", [False, True])
    max_depth_val = None if max_depth_is_none else trial.suggest_int("client.MAX_DEPTH", 2, 60)

    return {
        "MODEL_TYPE": "extratree",
        "N_ESTIMATORS": trial.suggest_int("client.N_ESTIMATORS", 50, 1200),
        "MAX_DEPTH": max_depth_val,
        "MAX_DEPTH_IS_NONE": max_depth_is_none,
        "MAX_FEATURES": trial.suggest_categorical(
            "client.MAX_FEATURES", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]
        ),
        "MIN_SAMPLES_SPLIT": trial.suggest_int("client.MIN_SAMPLES_SPLIT", 2, 30),
        "MIN_SAMPLES_LEAF": trial.suggest_int("client.MIN_SAMPLES_LEAF", 1, 30),
        "BOOTSTRAP": trial.suggest_categorical("client.BOOTSTRAP", [False, True]),
        "CRITERION": trial.suggest_categorical(
            "client.CRITERION", ["squared_error", "absolute_error"]
        ),
    }


def _mlp_space(trial) -> Dict[str, Any]:
    local_epochs = trial.suggest_int("client.LOCAL_EPOCHS", 2, 10)
    return {
        "MODEL_TYPE": "mlp",
        "LOCAL_EPOCHS": local_epochs,
        "BATCH_SIZE": trial.suggest_categorical("client.BATCH_SIZE", [16, 32, 64]),
        "LR": trial.suggest_float("client.LR", 1e-4, 5e-3, log=True),
        "WEIGHT_DECAY": trial.suggest_float("client.WEIGHT_DECAY", 1e-6, 5e-3, log=True),
        "DROPOUT": trial.suggest_float("client.DROPOUT", 0.0, 0.5),

        # --- beta ---
        "BETA": trial.suggest_float("client.BETA", 1e-2, 5.0, log=True),

        # --- early stopping ---
        "ES_ENABLED": trial.suggest_categorical("client.ES_ENABLED", [False, True]),
        "ES_VAL_FRAC": trial.suggest_float("client.ES_VAL_FRAC", 0.10, 0.25),
        "ES_MIN_EPOCHS": trial.suggest_int("client.ES_MIN_EPOCHS", 0, min(4, local_epochs - 1)),
        "ES_PATIENCE": trial.suggest_int("client.ES_PATIENCE", 1, min(6, local_epochs - 1)),
        "ES_MIN_DELTA": trial.suggest_float("client.ES_MIN_DELTA", 1e-5, 1e-2, log=True),
        "ES_RESTORE_BEST": trial.suggest_categorical("client.ES_RESTORE_BEST", [False, True]),


        "BEST_MODEL": trial.suggest_categorical("client.BEST_MODEL", [False, True]),

    }



def _tabnet_space(trial) -> Dict[str, Any]:
    """
    Spazio tipico per TabNetRegressor (pytorch-tabnet).
    Oltre ai parametri training (LR, WD, batch, epochs), ottimizziamo anche:
      - n_d / n_a (dim embedding decision/attention)
      - n_steps
      - gamma
      - n_independent / n_shared
      - lambda_sparse
      - mask_type
      - virtual_batch_size
      - momentum (BN momentum usato internamente)
    """
    n_d = trial.suggest_int("client.TABNET_N_D", 8, 64, log=True)
    # spesso si tiene n_a uguale a n_d; qui lo lasciamo ottimizzabile ma “vicino”
    n_a = trial.suggest_int("client.TABNET_N_A", 8, 64, log=True)

    return {
        "MODEL_TYPE": "TabNet",

        # training (coerenti con il tuo pattern client)
        "LOCAL_EPOCHS": trial.suggest_int("client.LOCAL_EPOCHS", 1, 10),
        "BATCH_SIZE": trial.suggest_categorical("client.BATCH_SIZE", [256, 512, 1024, 2048]),
        "LR": trial.suggest_float("client.LR", 5e-4, 5e-2, log=True),
        "WEIGHT_DECAY": trial.suggest_float("client.WEIGHT_DECAY", 1e-8, 1e-3, log=True),

        # architettura TabNet
        "TABNET_N_D": n_d,
        "TABNET_N_A": n_a,
        "TABNET_N_STEPS": trial.suggest_int("client.TABNET_N_STEPS", 3, 10),
        "TABNET_GAMMA": trial.suggest_float("client.TABNET_GAMMA", 1.0, 2.0),
        "TABNET_N_INDEPENDENT": trial.suggest_int("client.TABNET_N_INDEPENDENT", 1, 5),
        "TABNET_N_SHARED": trial.suggest_int("client.TABNET_N_SHARED", 1, 5),
        "TABNET_LAMBDA_SPARSE": trial.suggest_float("client.TABNET_LAMBDA_SPARSE", 1e-6, 1e-2, log=True),
        "TABNET_MASK_TYPE": trial.suggest_categorical("client.TABNET_MASK_TYPE", ["sparsemax", "entmax"]),

        # stabilità / batch norm
        "TABNET_VIRTUAL_BATCH_SIZE": trial.suggest_categorical("client.TABNET_VIRTUAL_BATCH_SIZE", [64, 128, 256]),
        "TABNET_MOMENTUM": trial.suggest_float("client.TABNET_MOMENTUM", 0.01, 0.4),
    }


def suggest_params(trial) -> Dict[str, Any]:
    """
    IMPORTANTISSIMO:
      Lo studio driver deve fare: trial.set_user_attr("model", args.model)
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
        client = _mlp_space(trial)
    elif model == "TabNet":
        client = _tabnet_space(trial)
    else:
        raise ValueError(f"Modello non supportato nello search space: {model}")

    return {"server": server, "client": client}
