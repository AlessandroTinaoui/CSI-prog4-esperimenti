import json
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import flwr as fl
from sklearn.metrics import mean_absolute_error

from config import HOLDOUT_CID, NUM_ROUNDS, SERVER_ADDRESS, TOP_K_FEATURES, MAX_MODELS_IN_ENSEMBLE
from strategy import ExtraTreesEnsembleStrategy

from dataset.dataset_cfg import get_train_path, get_test_path

TRAIN_PATH = get_train_path()
TEST_PATH = get_test_path()

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # xgboostmodel/
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = Path(__file__).resolve().parent.parent


def _load_global_ensemble(global_model_path: Path):
    raw = global_model_path.read_bytes()
    return pickle.loads(raw)


def main():
    global_model_path = RESULTS_DIR / "global_model.pkl"
    global_features_path = RESULTS_DIR / "global_model_features.json"

    for p in [global_model_path, global_features_path]:
        if p.exists():
            p.unlink()

    strategy = ExtraTreesEnsembleStrategy(
        top_k=TOP_K_FEATURES,
        save_path="selected_features.json",
        global_model_path="global_model.pkl",
        max_models_in_ensemble=MAX_MODELS_IN_ENSEMBLE,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=8,
    )

    print("🚀 Avvio Server Flower (ExtraTrees)...")
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

    if not global_model_path.exists() or not global_features_path.exists():
        print("❌ ERRORE: file global_model.pkl / global_model_features.json non creati.")
        sys.exit(1)

    ensemble = _load_global_ensemble(global_model_path)
    print("✅ Ensemble globale caricato.")

    train_features = json.loads(global_features_path.read_text(encoding="utf-8"))["features"]

    # -------------------------
    # GLOBAL TEST su HOLDOUT
    # -------------------------
    if HOLDOUT_CID <= 8:
        holdout_path = BASE_DIR / "../" / TRAIN_PATH / f"group{HOLDOUT_CID}_merged_clean.csv"
        if holdout_path.exists():
            holdout = pd.read_csv(holdout_path, sep=",").dropna()
            y_holdout = holdout["label"].copy()

            cols_to_drop = ["day", "client_id", "user_id", "source_file", "label"]
            cols_to_drop = [c for c in cols_to_drop if c in holdout.columns]
            X_holdout = holdout.drop(columns=cols_to_drop)

            # allinea feature
            for c in train_features:
                if c not in X_holdout.columns:
                    X_holdout[c] = 0
            X_holdout = X_holdout[train_features].fillna(0)

            y_pred_holdout = ensemble.predict(X_holdout)
            mae_holdout = mean_absolute_error(y_holdout, y_pred_holdout)
            print(f"📊 GLOBAL HOLDOUT MAE (client {HOLDOUT_CID}): {mae_holdout:.4f}")
        else:
            print(f"⚠️ Holdout non trovato: {holdout_path}")

    # -------------------------
    # TEST FINALE SU x_test_clean.csv
    # -------------------------
    test_path = BASE_DIR / "../" / TEST_PATH
    if not test_path.exists():
        print(f"⚠️ File x_test_clean.csv non trovato in {test_path}")
        return

    x_test = pd.read_csv(test_path)

    # ID (se presente) altrimenti indice
    if "id" in x_test.columns:
        ids = pd.to_numeric(x_test["id"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        ids = np.arange(len(x_test), dtype=int)

    X = x_test.drop(columns=[c for c in ["id", "label", "date"] if c in x_test.columns], errors="ignore")

    # allinea alle feature selezionate durante FL
    for c in train_features:
        if c not in X.columns:
            X[c] = 0
    X = X[train_features].fillna(0)

    y_pred = ensemble.predict(X)

    y_pred_int = np.rint(y_pred).astype(int)
    out = pd.DataFrame({"id": ids, "label": y_pred_int})
    out.to_csv("../results/predictions.csv", index=False)
    print("✅ Creato predictions.csv")


if __name__ == "__main__":
    main()
