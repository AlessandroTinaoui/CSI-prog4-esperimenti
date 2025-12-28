import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import flwr as fl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from config import HOLDOUT_CID
from config import (
    NUM_ROUNDS,
    SERVER_ADDRESS,
    TOP_K_FEATURES,
    N_BINS,
    HUBER_DELTA,
    REG_LAMBDA,
    GAMMA,
    LEARNING_RATE,
    BASE_SCORE,
)
from strategy import FederatedHistStumpStrategy




from dataset.dataset_cfg import get_train_path, get_test_path
TRAIN_PATH = get_train_path()
TEST_PATH = get_test_path()
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # xgboostmodel/
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = Path(__file__).resolve().parent.parent

def load_semicolon_csv_keep_rows(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"⚠️ File non trovato: {path}")
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    if not lines:
        return pd.DataFrame()

    header_line = lines[0].lstrip("\ufeff")
    header = [h.strip() for h in header_line.split(";")]
    ncol = len(header)
    fixed_lines = [";".join(header)]

    for line in lines[1:]:
        if line is None:
            line = ""
        parts = line.split(";")
        if len(parts) < ncol:
            parts = parts + [""] * (ncol - len(parts))
        elif len(parts) > ncol:
            parts = parts[: ncol - 1] + [";".join(parts[ncol - 1 :])]
        fixed_lines.append(";".join(parts))

    text = "\n".join(fixed_lines)
    return pd.read_csv(io.StringIO(text), sep=";", engine="python")


def clean_dataframe_soft(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _load_global_booster(global_model_path: Path) -> xgb.Booster:
    raw = global_model_path.read_bytes()
    bst = xgb.Booster()
    bst.load_model(bytearray(raw))
    return bst
def predict_stumps(X: pd.DataFrame, model_dict: dict) -> np.ndarray:
    pred = np.full((len(X),), float(model_dict.get("base_score", 0.0)), dtype=float)
    for s in model_dict.get("stumps", []) or []:
        feat = s["feature"]
        thr = float(s["thr"])
        wL = float(s["w_left"])
        wR = float(s["w_right"])
        lr = float(s.get("lr", 1.0))
        xcol = X[feat].to_numpy(dtype=float, copy=False)
        pred += lr * np.where(xcol <= thr, wL, wR)
    return pred



def main():
    global_model_path = RESULTS_DIR / "global_model.json"
    global_features_path = RESULTS_DIR / "global_model_features.json"

    for p in [global_model_path, global_features_path]:
        if p.exists():
            p.unlink()

    strategy = FederatedHistStumpStrategy(
        top_k=TOP_K_FEATURES,
        n_bins=N_BINS,
        huber_delta=HUBER_DELTA,
        reg_lambda=REG_LAMBDA,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        base_score=BASE_SCORE,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=8,
    )

    print("🚀 Avvio Server Flower...")
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
        print(global_model_path)
        print(global_features_path)
        print("❌ ERRORE: file global_model.json / global_model_features.json non creati.")
        sys.exit(1)

    model_dict = json.loads(global_model_path.read_text(encoding="utf-8"))
    print("✅ Modello globale (stumps) caricato.")

    train_features = json.loads(global_features_path.read_text(encoding="utf-8"))["features"]

    # -------------------------
    # GLOBAL TEST su HOLDOUT
    # -------------------------
    if HOLDOUT_CID <= 8:
        holdout_path = BASE_DIR / "../"/ TRAIN_PATH / f"group{HOLDOUT_CID}_merged_clean.csv"
        if holdout_path.exists():
            holdout = pd.read_csv(holdout_path, sep=",").dropna()
            y_holdout = holdout["label"].copy()

            cols_to_drop = ["day", "client_id", "user_id", "source_file", "label"]
            cols_to_drop = [c for c in cols_to_drop if c in holdout.columns]
            X_holdout = holdout.drop(columns=cols_to_drop)

            for c in train_features:
                if c not in X_holdout.columns:
                    X_holdout[c] = 0
            X_holdout = X_holdout[train_features].fillna(0)

            X_holdout = X_holdout[train_features].fillna(0)
            y_pred_holdout = predict_stumps(X_holdout, model_dict)
            mae_holdout = mean_absolute_error(y_holdout, y_pred_holdout)
            print(f"MEA valutato sul client {HOLDOUT_CID}")
            print(f"FINAL_MAE: {mae_holdout}")
        else:
            print(f"⚠️ Holdout non trovato: {holdout_path}")

    # -------------------------
    # TEST FINALE SU x_test_clean.csv (già pulito)
    # -------------------------
    test_path = BASE_DIR / "../" / TEST_PATH
    if not test_path.exists():
        print(f"⚠️ File x_test_clean.csv non trovato in {test_path}")
        return

    # Se è già pulito e con separatore standard:
    x_test = pd.read_csv(test_path)

    # ID (se presente) altrimenti indice
    if "id" in x_test.columns:
        ids = pd.to_numeric(x_test["id"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        ids = np.arange(len(x_test), dtype=int)

    # Feature matrix
    X = x_test.drop(columns=[c for c in ["id", "label", "date"] if c in x_test.columns], errors="ignore")

    # Allinea alle feature selezionate durante FL
    for c in train_features:
        if c not in X.columns:
            X[c] = 0
    X = X[train_features]

    # Allinea alle feature effettive del booster (se presenti)
    X = X[train_features].fillna(0)
    y_pred = predict_stumps(X, model_dict)

    y_pred_int = np.rint(y_pred).astype(int)
    out = pd.DataFrame({"id": ids, "label": y_pred_int})
    out.to_csv("../results/predictions.csv", index=False)
    print("✅ Creato predictions.csv")


if __name__ == "__main__":
    main()
