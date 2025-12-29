# mlp/client/config.py
from __future__ import annotations

import json
import os
from pathlib import Path

# --- DATA PIPELINE (FISSI) ---
CLIP_MIN = 0.0
CLIP_MAX = 100.0

TEST_SIZE = 0.2
RANDOM_STATE = 42
SHUFFLE_SPLIT = False

# --- TRAINING DEFAULTS (overridabili da trial) ---
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
LR = 0.0006227096731372952
WEIGHT_DECAY = 0.00030457842752558905

# --- MODEL DEFAULTS (overridabili da trial) ---
HIDDEN_SIZES = [64, 32, 16]
DROPOUT = 0.3516229788661295
BETA=2.0

def _apply_trial_overrides() -> None:
    global LOCAL_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY
    global HIDDEN_SIZES, DROPOUT, BETA

    cfg_path = os.environ.get("TRIAL_CONFIG_PATH")
    if not cfg_path:
        return
    p = Path(cfg_path)
    if not p.exists():
        return

    cfg = json.loads(p.read_text(encoding="utf-8"))
    client = cfg.get("client", {})

    if "LOCAL_EPOCHS" in client:
        LOCAL_EPOCHS = int(client["LOCAL_EPOCHS"])
    if "BATCH_SIZE" in client:
        BATCH_SIZE = int(client["BATCH_SIZE"])
    if "LR" in client:
        LR = float(client["LR"])
    if "WEIGHT_DECAY" in client:
        WEIGHT_DECAY = float(client["WEIGHT_DECAY"])

    if "HIDDEN_SIZES" in client:
        HIDDEN_SIZES = list(client["HIDDEN_SIZES"])
    if "DROPOUT" in client:
        DROPOUT = float(client["DROPOUT"])
    if "BETA" in client:
        BETA = float(client["BETA"])


_apply_trial_overrides()
