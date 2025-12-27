from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import flwr as fl
import torch
from sklearn.metrics import mean_absolute_error

from config import (
    HOLDOUT_CID, NUM_ROUNDS, SERVER_ADDRESS,
    RESULTS_DIRNAME, GLOBAL_FEATURES_JSON, GLOBAL_SCALER_JSON,
    MIN_AVAILABLE_CLIENTS, MIN_EVALUATE_CLIENTS, MIN_FIT_CLIENTS,
    FRACTION_EVALUATE, FRACTION_FIT
)
from strategy import FedProxNNWithGlobalScaler
from TabNet.model import TabNetRegressor, TabNetConfig
import TabNet.client.client_params as P

from dataset.dataset_cfg import get_train_path, get_test_path


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / RESULTS_DIRNAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent


def _load_global_artifacts():
    features_path = RESULTS_DIR / GLOBAL_FEATURES_JSON
    scaler_path = RESULTS_DIR / GLOBAL_SCALER_JSON
    model_path = RESULTS_DIR / "global_model.npz"
    target_path = RESULTS_DIR / "global_target.json"

    for p in [features_path, scaler_path, model_path, target_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    global_features = json.loads(features_path.read_text(encoding="utf-8"))["features"]

    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)

    tgt = json.loads(target_path.read_text(encoding="utf-8"))
    y_mean = float(tgt.get("y_mean", 0.0))
    y_std = float(tgt.get("y_std", 1.0))
    if y_std <= 1e-12:
        y_std = 1.0

    npz = np.load(model_path)
    params = [npz[f"arr_{i}"] for i in range(len(npz.files))]

    return global_features, mean, std, params, y_mean, y_std


def _align_and_standardize(X: pd.DataFrame, global_features, mean, std) -> np.ndarray:
    for c in global_features:
        if c not in X.columns:
            X[c] = 0.0
    X = X[global_features].fillna(0.0).to_numpy(dtype=np.float32)

    std_safe = np.where(std <= 1e-12, 1.0, std)
    X = (X - mean) / std_safe
    return X


def _set_torch_params(model: torch.nn.Module, params):
    state = model.state_dict()
    keys = list(state.keys())
    if len(keys) != len(params):
        raise ValueError(f"Param mismatch: got {len(params)}, expected {len(keys)}")
    new_state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(new_state, strict=True)


def _predict_real(model: torch.nn.Module, X: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        pred_norm = model(xb).cpu().numpy()

    pred = pred_norm * y_std + y_mean
    pred = np.clip(pred, 0.0, 100.0)
    return pred


def main():
    strategy = FedProxNNWithGlobalScaler(
        project_root=PROJECT_ROOT,
        fedprox_mu=float(P.FEDPROX_MU),
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
    )

    print("Avvio Server Flower (TabNet) con FedProx...")
    try:
        fl.server.start_server(
            server_address=SERVER_ADDRESS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
        )
    except Exception as e:
        print(f"Errore durante il training FL: {e}")
        sys.exit(1)

    print("\nFL terminato. Inizio fase di test...")

    try:
        global_features, mean, std, params, y_mean, y_std = _load_global_artifacts()
    except Exception as e:
        print(f"ERRORE artifacts globali: {e}")
        sys.exit(1)

    print(f"Artifacts caricati: n_features={len(global_features)} | y_mean={y_mean:.4f} y_std={y_std:.4f}")

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
    _set_torch_params(model, params)

    # 1) HOLDOUT MAE
    train_dir = Path(get_train_path())
    if 0 <= HOLDOUT_CID <= 8:
        holdout_path = (BASE_DIR / "../.." / train_dir / f"group{HOLDOUT_CID}_merged_clean.csv").resolve()
        if holdout_path.exists():
            holdout = pd.read_csv(holdout_path, sep=",")
            if "label" in holdout.columns:
                y_holdout = holdout["label"].astype(float).to_numpy()
                cols_to_drop = [c for c in ["day", "client_id", "user_id", "source_file", "label"] if c in holdout.columns]
                X_holdout = holdout.drop(columns=cols_to_drop, errors="ignore")
                Xh = _align_and_standardize(X_holdout, global_features, mean, std)
                y_pred_holdout = _predict_real(model, Xh, y_mean, y_std)
                mae_holdout = mean_absolute_error(y_holdout, y_pred_holdout)
                print(f"HOLDOUT client {HOLDOUT_CID} - FINAL_MAE: {mae_holdout}")
            else:
                print(f"Holdout senza label: {holdout_path}")
        else:
            print(f"Holdout non trovato: {holdout_path}")

    # 2) PREDICT test Kaggle
    test_path = (BASE_DIR / "../.." / Path(get_test_path())).resolve()
    if not test_path.exists():
        print(f"File test non trovato in {test_path}")
        return

    x_test = pd.read_csv(test_path)

    if "id" in x_test.columns:
        ids = pd.to_numeric(x_test["id"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        ids = np.arange(len(x_test), dtype=int)

    X = x_test.drop(columns=[c for c in ["id", "label", "date"] if c in x_test.columns], errors="ignore")
    Xt = _align_and_standardize(X, global_features, mean, std)
    y_pred = _predict_real(model, Xt, y_mean, y_std).astype(np.float32)

    sub = pd.DataFrame({"id": ids, "label": y_pred})
    out_csv = RESULTS_DIR / "submission.csv"
    sub.to_csv(out_csv, index=False)
    print(f"Submission salvata in: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
