# mlp/server/config.py
from __future__ import annotations

import json
import os
from pathlib import Path

# --- networking / fl ---
SERVER_ADDRESS = "127.0.0.1:8080"

if "FL_SERVER_ADDRESS" in os.environ:
    SERVER_ADDRESS = os.environ["FL_SERVER_ADDRESS"]
HOLDOUT_CID = 2
# --- default federated params (overridabili da trial) ---
NUM_ROUNDS = 87
FRACTION_FIT = 1.0
FRACTION_EVALUATE = 1.0
MIN_FIT_CLIENTS = 8
MIN_EVALUATE_CLIENTS = 8
MIN_AVAILABLE_CLIENTS = 8
TOP_K_FEATURES = 30
MIN_STD_FEATURE_FS = 1e-8


# --- output ---
RESULTS_DIRNAME = "results"
GLOBAL_FEATURES_JSON = "global_features.json"
GLOBAL_SCALER_JSON = "global_scaler.json"
GLOBAL_MODEL_PTH = "global_model.pth"


def _apply_trial_overrides() -> None:
    global NUM_ROUNDS, FRACTION_FIT, FRACTION_EVALUATE
    global MIN_FIT_CLIENTS, MIN_EVALUATE_CLIENTS, MIN_AVAILABLE_CLIENTS

    cfg_path = os.environ.get("TRIAL_CONFIG_PATH")
    if not cfg_path:
        return
    p = Path(cfg_path)
    if not p.exists():
        return

    cfg = json.loads(p.read_text(encoding="utf-8"))
    server = cfg.get("server", {})

    if "NUM_ROUNDS" in server:
        NUM_ROUNDS = int(server["NUM_ROUNDS"])

    if "FRACTION_FIT" in server:
        FRACTION_FIT = float(server["FRACTION_FIT"])
    if "FRACTION_EVALUATE" in server:
        FRACTION_EVALUATE = float(server["FRACTION_EVALUATE"])

    if "MIN_FIT_CLIENTS" in server:
        MIN_FIT_CLIENTS = int(server["MIN_FIT_CLIENTS"])
    if "MIN_EVALUATE_CLIENTS" in server:
        MIN_EVALUATE_CLIENTS = int(server["MIN_EVALUATE_CLIENTS"])
    if "MIN_AVAILABLE_CLIENTS" in server:
        MIN_AVAILABLE_CLIENTS = int(server["MIN_AVAILABLE_CLIENTS"])


_apply_trial_overrides()
