import json
import os
from pathlib import Path

SERVER_ADDRESS = "localhost:8080"

# Default
NUM_ROUNDS = 10
HOLDOUT_CID = 2
TOP_K_FEATURES = 30
LOCAL_BOOST_ROUND = 1


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
    global NUM_ROUNDS, HOLDOUT_CID, TOP_K_FEATURES, LOCAL_BOOST_ROUND

    cfg = _load_trial_config()
    server_cfg = cfg.get("server", {}) if isinstance(cfg, dict) else {}

    # Override da JSON
    if "NUM_ROUNDS" in server_cfg:
        NUM_ROUNDS = int(server_cfg["NUM_ROUNDS"])
    if "HOLDOUT_CID" in server_cfg:
        HOLDOUT_CID = int(server_cfg["HOLDOUT_CID"])
    if "TOP_K_FEATURES" in server_cfg:
        TOP_K_FEATURES = int(server_cfg["TOP_K_FEATURES"])
    if "LOCAL_BOOST_ROUND" in server_cfg:
        LOCAL_BOOST_ROUND = int(server_cfg["LOCAL_BOOST_ROUND"])

    # Override "più forte" da env (utile per holdout loop senza riscrivere JSON)
    env_holdout = os.environ.get("HOLDOUT_CID", "").strip()
    if env_holdout != "":
        try:
            HOLDOUT_CID = int(env_holdout)
        except ValueError:
            pass


_apply_overrides()
