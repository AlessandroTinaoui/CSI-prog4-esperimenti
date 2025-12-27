import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from TabNet.model import TabNetRegressor, TabNetConfig
import TabNet.client.client_params as P
from dataset.dataset_cfg import get_train_path, get_test_path

def load_artifacts(results_dir: Path):
    global_features = json.loads((results_dir / "global_features.json").read_text(encoding="utf-8"))["features"]

    scaler = json.loads((results_dir / "global_scaler.json").read_text(encoding="utf-8"))
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)

    tgt = json.loads((results_dir / "global_target.json").read_text(encoding="utf-8"))
    y_mean = float(tgt.get("y_mean", 0.0))
    y_std = float(tgt.get("y_std", 1.0))
    if y_std <= 1e-12:
        y_std = 1.0

    npz = np.load(results_dir / "global_model.npz")
    params = [npz[f"arr_{i}"] for i in range(len(npz.files))]

    return global_features, mean, std, y_mean, y_std, params


def align_and_standardize(dfX: pd.DataFrame, global_features, mean, std):
    for c in global_features:
        if c not in dfX.columns:
            dfX[c] = 0.0
    X = dfX[global_features].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    std_safe = np.where(std <= 1e-12, 1.0, std).astype(np.float32)
    X = (X - mean) / std_safe
    return X


def set_torch_params(model: torch.nn.Module, params):
    state = model.state_dict()
    keys = list(state.keys())
    if len(keys) != len(params):
        raise ValueError(f"Param mismatch: got {len(params)}, expected {len(keys)}")
    new_state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(new_state, strict=True)


def predict_real(model, X, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        pred_norm = model(xb).cpu().numpy()
    pred = pred_norm * y_std + y_mean
    return pred  # NO CLIP


def describe_y(y: np.ndarray):
    q = np.percentile(y, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "min": float(q[0]),
        "p01": float(q[1]),
        "p05": float(q[2]),
        "p25": float(q[3]),
        "p50": float(q[4]),
        "p75": float(q[5]),
        "p95": float(q[6]),
        "p99": float(q[7]),
        "max": float(q[8]),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
    }


def feature_shift_stats(X_std: np.ndarray):
    # Dopo standardizzazione globale, se tutto è "in distribuzione", mean~0, std~1.
    # Qui misuriamo quanto si discosta.
    m = np.mean(X_std, axis=0)
    s = np.std(X_std, axis=0)
    return float(np.mean(np.abs(m))), float(np.mean(np.abs(s - 1.0)))

PATH_TO_TRAIN_DIR = get_train_path()
def main():
    # Adatta questi path se necessario
    PROJECT_ROOT = Path(__file__).resolve().parent
    RESULTS_DIR = PROJECT_ROOT / "server/results"

    # DATA_DIR: metti qui la cartella dove stanno group0_merged_clean.csv ... group8_merged_clean.csv
    # Se usi la stessa di run_all.py, punta lì.
    DATA_DIR = (PROJECT_ROOT  / ".." / PATH_TO_TRAIN_DIR).resolve()

    global_features, mean, std, y_mean, y_std, params = load_artifacts(RESULTS_DIR)

    tabcfg = TabNetConfig(
        n_d=P.TABNET_N_D,
        n_a=P.TABNET_N_A,
        n_steps=P.TABNET_N_STEPS,
        gamma=P.TABNET_GAMMA,
        n_shared=P.TABNET_N_SHARED,
        n_independent=P.TABNET_N_INDEPENDENT,
        bn_virtual_bs=P.TABNET_BN_VIRTUAL_BS,
        bn_momentum=P.TABNET_BN_MOMENTUM,
    )
    model = TabNetRegressor(input_dim=len(global_features), cfg=tabcfg)
    set_torch_params(model, params)

    rows = []
    for cid in range(9):
        path = DATA_DIR / f"group{cid}_merged_clean.csv"
        if not path.exists():
            print(f"[SKIP] Missing: {path}")
            continue

        df = pd.read_csv(path)
        if "label" not in df.columns:
            print(f"[SKIP] No label in: {path}")
            continue

        y = df["label"].astype(float).to_numpy()
        drop_cols = [c for c in ["day", "client_id", "user_id", "source_file", "label"] if c in df.columns]
        Xdf = df.drop(columns=drop_cols, errors="ignore")

        Xstd = align_and_standardize(Xdf, global_features, mean, std)
        pred = predict_real(model, Xstd, y_mean, y_std)

        mae = mean_absolute_error(y, pred)

        ydesc = describe_y(y)
        mean_abs_feat_mean, mean_abs_feat_std_err = feature_shift_stats(Xstd)

        rows.append({
            "client": cid,
            "n": len(df),
            "MAE": float(mae),
            "y_mean": ydesc["mean"],
            "y_std": ydesc["std"],
            "y_min": ydesc["min"],
            "y_p05": ydesc["p05"],
            "y_p50": ydesc["p50"],
            "y_p95": ydesc["p95"],
            "y_max": ydesc["max"],
            "feat_shift_mean_abs": mean_abs_feat_mean,
            "feat_shift_std_abs_err": mean_abs_feat_std_err,
        })

    out = pd.DataFrame(rows).sort_values("MAE", ascending=False)
    print("\n=== Per-client summary (sorted by MAE desc) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
