import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import flwr as fl
import torch
from sklearn.metrics import mean_absolute_error

from config import (
    HOLDOUT_CID, NUM_ROUNDS, SERVER_ADDRESS,
    RESULTS_DIRNAME, GLOBAL_FEATURES_JSON, GLOBAL_SCALER_JSON
)
from strategy import FedAvgNNWithGlobalScaler

# Se nel tuo progetto hai già questi metodi, li riuso come fai ora :contentReference[oaicite:3]{index=3}
from dataset.dataset_cfg import get_train_path, get_test_path

from nnmodel.model import MLPRegressor


TRAIN_PATH = get_train_path()
TEST_PATH = get_test_path()

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # nnmodel/
RESULTS_DIR = PROJECT_ROOT / RESULTS_DIRNAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent.parent  # coerente al tuo server_flwr :contentReference[oaicite:4]{index=4}


def _load_global_artifacts():
    features_path = RESULTS_DIR / GLOBAL_FEATURES_JSON
    scaler_path = RESULTS_DIR / GLOBAL_SCALER_JSON
    model_path = RESULTS_DIR / "global_model.npz"

    for p in [features_path, scaler_path, model_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    global_features = json.loads(features_path.read_text(encoding="utf-8"))["features"]
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)

    npz = np.load(model_path)
    params = [npz[f"arr_{i}"] for i in range(len(npz.files))]

    return global_features, mean, std, params


def _align_and_standardize(X: pd.DataFrame, global_features, mean, std) -> np.ndarray:
    # allinea feature come fai tu per booster: colonne mancanti a 0 e ordine fisso :contentReference[oaicite:5]{index=5}
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


def _predict(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        pred = model(xb).cpu().numpy()
    # clip score Garmin tipico 0..100 (come protezione)
    pred = np.clip(pred, 0.0, 100.0)
    return pred


def main():
    # Strategy NN (la tua FedAvg con scaler global round 1)
    strategy = FedAvgNNWithGlobalScaler(
        project_root=PROJECT_ROOT,
        #num_rounds=NUM_ROUNDS,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=8,
    )

    print("🚀 Avvio Server Flower (NN)...")
    try:
        fl.server.start_server(
            server_address=SERVER_ADDRESS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
        )
    except Exception as e:
        print(f"❌ Errore durante il training FL: {e}")
        sys.exit(1)

    print("\n✅ FL terminato. Inizio fase di test...")

    # Carica artifacts finali (features/scaler/model)
    try:
        global_features, mean, std, params = _load_global_artifacts()
    except Exception as e:
        print(f"❌ ERRORE artifacts globali: {e}")
        sys.exit(1)

    print(f"✅ Artifacts caricati: n_features={len(global_features)}")

    # Ricostruisci modello con input_dim corretto
    input_dim = len(global_features)
    model = MLPRegressor(input_dim=input_dim, hidden_sizes=[64, 32, 16], dropout=0.0)
    _set_torch_params(model, params)

    # -------------------------
    # 1) GLOBAL TEST su HOLDOUT
    # -------------------------
    if HOLDOUT_CID <= 8:
        holdout_path = BASE_DIR / "../" / TRAIN_PATH / f"group{HOLDOUT_CID}_merged_clean.csv"
        if holdout_path.exists():
            holdout = pd.read_csv(holdout_path, sep=",").dropna()

            if "label" not in holdout.columns:
                print(f"⚠️ Holdout senza label: {holdout_path}")
            else:
                y_holdout = holdout["label"].astype(float).to_numpy()

                cols_to_drop = ["day", "client_id", "user_id", "source_file", "label"]
                cols_to_drop = [c for c in cols_to_drop if c in holdout.columns]
                X_holdout = holdout.drop(columns=cols_to_drop, errors="ignore")

                # Allinea + standardizza con scaler globale
                Xh = _align_and_standardize(X_holdout, global_features, mean, std)

                y_pred_holdout = _predict(model, Xh)
                mae_holdout = mean_absolute_error(y_holdout, y_pred_holdout)

                print(f"MAE valutato sul client holdout {HOLDOUT_CID}")
                print(f"FINAL_HOLDOUT_MAE: {mae_holdout}")
        else:
            print(f"⚠️ Holdout non trovato: {holdout_path}")

    # -------------------------
    # 2) TEST FINALE su x_test_clean.csv (già pulito)
    # -------------------------
    test_path = BASE_DIR / "../" / TEST_PATH
    if not test_path.exists():
        print(f"⚠️ File x_test_clean.csv non trovato in {test_path}")
        return

    x_test = pd.read_csv(test_path)

    # ID come fai tu: se non c'è, indice :contentReference[oaicite:6]{index=6}
    if "id" in x_test.columns:
        ids = pd.to_numeric(x_test["id"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        ids = np.arange(len(x_test), dtype=int)

    # Feature matrix: togli id/label/date se presenti (stesso stile) :contentReference[oaicite:7]{index=7}
    X = x_test.drop(columns=[c for c in ["id", "label", "date"] if c in x_test.columns], errors="ignore")

    Xt = _align_and_standardize(X, global_features, mean, std)
    y_pred = _predict(model, Xt)

    # output in int come fai tu (rint) :contentReference[oaicite:8]{index=8}
    y_pred_int = np.rint(y_pred).astype(int)
    out = pd.DataFrame({"id": ids, "label": y_pred_int})
    out.to_csv("../results/predictions.csv", index=False)
    print("✅ Creato predictions.csv")


if __name__ == "__main__":
    main()
