import json
import os
from pathlib import Path

SERVER_ADDRESS = "localhost:8080"

# Default
NUM_ROUNDS = 171
HOLDOUT_CID = 10
TOP_K_FEATURES = 50


# --- Federated GBDT (stumps) ---
N_BINS = 128          # numero di bin per feature (quantili)
HUBER_DELTA = 2.2288150983938597    # delta per pseudo-huber (più basso -> più simile a MAE)
REG_LAMBDA = 3.8944557365876697     # reg L2 su leaf weight
GAMMA = 2.8772478825619396          # penalità per split (opzionale)
LEARNING_RATE = 0.2447102091138996  # shrinkage per ogni stump
BASE_SCORE = -3.450512078882384     # bias iniziale (puoi metterlo a media label se vuoi)


def _load_trial_config() -> dict:
    """
    Carica un JSON di configurazione se l'env TRIAL_CONFIG_PATH è impostata.
    Ritorna {} se non presente o se fallisce.
    """
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
    global NUM_ROUNDS, HOLDOUT_CID, TOP_K_FEATURES
    global N_BINS, HUBER_DELTA, REG_LAMBDA, GAMMA, LEARNING_RATE, BASE_SCORE
    cfg = _load_trial_config()
    server_cfg = cfg.get("server", {}) if isinstance(cfg, dict) else {}

    # Override da JSON
    if "NUM_ROUNDS" in server_cfg:
        NUM_ROUNDS = int(server_cfg["NUM_ROUNDS"])

    if "TOP_K_FEATURES" in server_cfg:
        TOP_K_FEATURES = int(server_cfg["TOP_K_FEATURES"])
    if "N_BINS" in server_cfg:
        N_BINS = int(server_cfg["N_BINS"])
    if "HUBER_DELTA" in server_cfg:
        HUBER_DELTA = float(server_cfg["HUBER_DELTA"])
    if "REG_LAMBDA" in server_cfg:
        REG_LAMBDA = float(server_cfg["REG_LAMBDA"])
    if "GAMMA" in server_cfg:
        GAMMA = float(server_cfg["GAMMA"])
    if "LEARNING_RATE" in server_cfg:
        LEARNING_RATE = float(server_cfg["LEARNING_RATE"])
    if "BASE_SCORE" in server_cfg:
        BASE_SCORE = float(server_cfg["BASE_SCORE"])




_apply_overrides()
