# config.py
from __future__ import annotations

import json
import os
from pathlib import Path

SERVER_ADDRESS = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8080")

# se vuoi holdout di un client: metti 0..8, altrimenti >=9 per disabilitare
HOLDOUT_CID = int(os.environ.get("HOLDOUT_CID", "0"))

NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "80"))
FRACTION_FIT = float(os.environ.get("FRACTION_FIT", "1.0"))
FRACTION_EVALUATE = float(os.environ.get("FRACTION_EVALUATE", "1.0"))
MIN_FIT_CLIENTS = int(os.environ.get("MIN_FIT_CLIENTS", "8"))
MIN_EVALUATE_CLIENTS = int(os.environ.get("MIN_EVALUATE_CLIENTS", "8"))
MIN_AVAILABLE_CLIENTS = int(os.environ.get("MIN_AVAILABLE_CLIENTS", "8"))

RESULTS_DIRNAME = "results"
GLOBAL_FEATURES_JSON = "global_features.json"
GLOBAL_SCALER_JSON = "global_scaler.json"


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
