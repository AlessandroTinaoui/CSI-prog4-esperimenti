import json
import os
from pathlib import Path

# ===== Default (fallback) =====
ET_N_ESTIMATORS = 750
ET_MAX_DEPTH = None
ET_MIN_SAMPLES_SPLIT = 18
ET_MIN_SAMPLES_LEAF = 1
ET_MAX_FEATURES = 1.0
ET_BOOTSTRAP = True
ET_CRITERION = "absolute_error"  # se lo vuoi supportare

# ===== Override da TRIAL_CONFIG_PATH (se presente) =====
_cfg_path = os.environ.get("TRIAL_CONFIG_PATH")
if _cfg_path:
    p = Path(_cfg_path)
    if p.exists():
        cfg = json.loads(p.read_text(encoding="utf-8"))
        client = cfg.get("client", {})

        # supporto sia chiavi "client.N_ESTIMATORS" sia "N_ESTIMATORS" (nel dubbio)
        ET_N_ESTIMAT,ORS = int(client.get("N_ESTIMATORS", client.get("client.N_ESTIMATORS", ET_N_ESTIMATORS)))

        # MAX_DEPTH: può essere None oppure un int
        # nel tuo search_space hai anche MAX_DEPTH_IS_NONE, quindi gestiscilo
        is_none = client.get("MAX_DEPTH_IS_NONE", client.get("client.MAX_DEPTH_IS_NONE", None))
        if is_none is True:
            ET_MAX_DEPTH = None
        else:
            ET_MAX_DEPTH = client.get("MAX_DEPTH", client.get("client.MAX_DEPTH", ET_MAX_DEPTH))
            if ET_MAX_DEPTH is not None:
                ET_MAX_DEPTH = int(ET_MAX_DEPTH)

        ET_MAX_FEATURES = client.get("MAX_FEATURES", client.get("client.MAX_FEATURES", ET_MAX_FEATURES))
        ET_MIN_SAMPLES_SPLIT = int(client.get("MIN_SAMPLES_SPLIT", client.get("client.MIN_SAMPLES_SPLIT", ET_MIN_SAMPLES_SPLIT)))
        ET_MIN_SAMPLES_LEAF = int(client.get("MIN_SAMPLES_LEAF", client.get("client.MIN_SAMPLES_LEAF", ET_MIN_SAMPLES_LEAF)))
        ET_BOOTSTRAP = bool(client.get("BOOTSTRAP", client.get("client.BOOTSTRAP", ET_BOOTSTRAP)))

        # opzionale
        ET_CRITERION = client.get("CRITERION", client.get("client.CRITERION", ET_CRITERION))

# ===== Params finali =====
EXTRA_TREES_PARAMS = {
    "n_estimators": ET_N_ESTIMATORS,
    "max_depth": ET_MAX_DEPTH,
    "max_features": ET_MAX_FEATURES,
    "bootstrap": ET_BOOTSTRAP,
    "min_samples_split": ET_MIN_SAMPLES_SPLIT,
    "min_samples_leaf": ET_MIN_SAMPLES_LEAF,
    # opzionale se lo usi davvero nel costruttore
    "criterion": ET_CRITERION,
}

print(
    "[EXTRATREES PARAMS]",
    f"TRIAL_CONFIG_PATH={_cfg_path!r}",
    EXTRA_TREES_PARAMS,
    flush=True,
)
