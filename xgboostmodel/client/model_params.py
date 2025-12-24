# model_params.py
# Model parameters (default + override da trial config)

import json
import os
from pathlib import Path

# Default
N_ESTIMATORS = 100
MAX_DEPTH = 5
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
REG_LAMBDA = 1.0
RANDOM_STATE = 42
N_JOBS = 1
OBJECTIVE = "reg:absoluteerror"
EVAL_METRIC = "mae"
VERBOSITY = 0

# Early stopping
MAX_LOCAL_ROUNDS = 50
ES_ROUNDS = 5
ES_DATA_NAME = "valid"
ES_MAXIMIZE = False
ES_SAVE_BEST = True


def _load_trial_config() -> dict:
    p = os.environ.get("TRIAL_CONFIG_PATH", "").strip()
    if not p:
        return {}
    try:
        cfg_path = Path(p).expanduser().resolve()
        if not cfg_path.exists():
            return {}
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def _apply_overrides():
    global N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE_BYTREE, REG_LAMBDA
    global RANDOM_STATE, N_JOBS, OBJECTIVE, EVAL_METRIC, VERBOSITY
    global MAX_LOCAL_ROUNDS, ES_ROUNDS, ES_DATA_NAME, ES_MAXIMIZE, ES_SAVE_BEST

    cfg = _load_trial_config()
    client_cfg = cfg.get("client", {}) if isinstance(cfg, dict) else {}

    # Parametri modello
    if "N_ESTIMATORS" in client_cfg:
        N_ESTIMATORS = int(client_cfg["N_ESTIMATORS"])
    if "MAX_DEPTH" in client_cfg:
        MAX_DEPTH = int(client_cfg["MAX_DEPTH"])
    if "LEARNING_RATE" in client_cfg:
        LEARNING_RATE = float(client_cfg["LEARNING_RATE"])
    if "SUBSAMPLE" in client_cfg:
        SUBSAMPLE = float(client_cfg["SUBSAMPLE"])
    if "COLSAMPLE_BYTREE" in client_cfg:
        COLSAMPLE_BYTREE = float(client_cfg["COLSAMPLE_BYTREE"])
    if "REG_LAMBDA" in client_cfg:
        REG_LAMBDA = float(client_cfg["REG_LAMBDA"])

    # Training control
    if "MAX_LOCAL_ROUNDS" in client_cfg:
        MAX_LOCAL_ROUNDS = int(client_cfg["MAX_LOCAL_ROUNDS"])
    if "ES_ROUNDS" in client_cfg:
        ES_ROUNDS = int(client_cfg["ES_ROUNDS"])


_apply_overrides()
