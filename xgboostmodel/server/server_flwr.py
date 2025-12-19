# server_flwr.py
from __future__ import annotations

import io
import json
import os
import pickle
import sys
from pathlib import Path

import flwr as fl
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------------------------
# Import preprocessing da dataset/CSV_train (client_dataset_setup.py)
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../CSI-prog4-esperimenti
PREP_DIR = PROJECT_ROOT / "dataset" / "CSV_train"
sys.path.insert(0, str(PREP_DIR))

from client_dataset_setup import CleanConfig, clean_user_df, read_user_csv  # noqa: E402

# -------------------------------------------------------------------
# Config + Strategy
# -------------------------------------------------------------------
from config import HOLDOUT_CID, NUM_ROUNDS, SERVER_ADDRESS  # noqa: E402
from strategy import XGBoostEnsembleAggregation  # noqa: E402


# -------------------------
# Utils
# -------------------------
def load_semicolon_csv_keep_rows(path: str) -> pd.DataFrame:
    """Carica CSV ';' in modo robusto senza perdere righe anche se ci sono ';' dentro celle."""
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
    """Pulizia 'soft': non droppa righe, solo sistemazioni colonne/inf."""
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def predict_ensemble(ensemble_items: list[dict], X: pd.DataFrame) -> np.ndarray:
    """Predizione con ensemble di booster, riallineando le feature per ogni booster."""
    preds = []
    weights = []

    for item in ensemble_items:
        booster = xgb.Booster()
        booster.load_model(bytearray(item["raw"]))

        model_feats = booster.feature_names or list(X.columns)

        # aggiungi feature mancanti (robustezza)
        for f in model_feats:
            if f not in X.columns:
                X[f] = 0

        X_aligned = X[model_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        dmat = xgb.DMatrix(X_aligned, feature_names=model_feats)

        preds.append(booster.predict(dmat))
        weights.append(float(item.get("weight", 1.0)))

    if len(preds) == 0:
        return np.array([])

    weights = np.array(weights, dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights, dtype=float)
    weights /= weights.sum()

    y_pred = np.average(np.vstack(preds), axis=0, weights=weights)
    return y_pred


# -------------------------
# Main
# -------------------------
def main():
    BASE_DIR = Path(__file__).resolve().parent

    # questi file li crea la tua strategy (quindi devono combaciare con la strategy)
    GLOBAL_ENSEMBLE_PATH = BASE_DIR / "global_ensemble.pkl"
    GLOBAL_FEATURES_PATH = BASE_DIR / "global_model_features.json"

    # pulisci vecchi file
    if GLOBAL_ENSEMBLE_PATH.exists():
        GLOBAL_ENSEMBLE_PATH.unlink()
    if GLOBAL_FEATURES_PATH.exists():
        GLOBAL_FEATURES_PATH.unlink()

    # Strategy
    strategy = XGBoostEnsembleAggregation(
        top_k=20,
        save_path="selected_features.json",
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

    # Verifica file prodotti
    if not GLOBAL_ENSEMBLE_PATH.exists() or not GLOBAL_FEATURES_PATH.exists():
        print("❌ ERRORE: I file del modello globale (pkl/json) non sono stati creati.")
        print("   Verifica che i client non siano crashati e che la Strategy abbia salvato i file.")
        sys.exit(1)

    # Carica ensemble
    with open(GLOBAL_ENSEMBLE_PATH, "rb") as f:
        ensemble_items = pickle.load(f)
    print(f"✅ Ensemble caricato: {len(ensemble_items)} booster.")

    # Carica features globali
    with open(GLOBAL_FEATURES_PATH, "r", encoding="utf-8") as f:
        train_features = json.load(f)["features"]
    train_features = list(train_features)

    # -------------------------
    # GLOBAL TEST su HOLDOUT
    # -------------------------
    if 0 <= HOLDOUT_CID <= 8:
        holdout_path = (BASE_DIR / ".." / "clients_data" / f"group{HOLDOUT_CID}_merged_clean.csv").resolve()
        print("[SERVER] Holdout path:", holdout_path)

        if not holdout_path.exists():
            print("⚠️ Holdout non trovato, salto MAE holdout.")
        else:
            df_holdout = pd.read_csv(holdout_path, sep=",")

            if "label" not in df_holdout.columns:
                print("⚠️ Holdout senza colonna label, salto MAE holdout.")
            else:
                # ✅ NON fare dropna() totale: al massimo richiedi label non NaN
                df_holdout = df_holdout.dropna(subset=["label"])

                cols_to_drop = ["day", "client_id", "user_id", "source_file", "label"]
                cols_to_drop = [c for c in cols_to_drop if c in df_holdout.columns]

                y_holdout = df_holdout["label"].to_numpy()
                X_holdout = df_holdout.drop(columns=cols_to_drop, errors="ignore")

                # allinea alle train_features globali
                for c in train_features:
                    if c not in X_holdout.columns:
                        X_holdout[c] = 0
                X_holdout = X_holdout[train_features].replace([np.inf, -np.inf], np.nan).fillna(0)

                print("[SERVER] Holdout rows:", len(df_holdout))
                print("[SERVER] X_holdout shape:", X_holdout.shape)
                print("[SERVER] y_holdout shape:", y_holdout.shape)

                y_pred_holdout = predict_ensemble(ensemble_items, X_holdout.copy())

                if len(y_holdout) == 0 or len(y_pred_holdout) == 0:
                    print("⚠️ HOLDOUT vuoto o predizioni vuote: salto MAE holdout.")
                else:
                    mae_holdout = mean_absolute_error(y_holdout, y_pred_holdout)
                    print(f"📊 GLOBAL HOLDOUT MAE (client {HOLDOUT_CID}): {mae_holdout:.4f}")

    # -------------------------
    # TEST FINALE SU x_test.csv
    # -------------------------
    test_path = (BASE_DIR / "x_test.csv").resolve()
    if not test_path.exists():
        print(f"❌ File x_test.csv non trovato in {test_path}")
        return

    # ✅ Preprocessing uguale al train ma "infer": NON deve droppare righe.
    # (Funziona solo se nel tuo client_dataset_setup.py hai implementato mode="infer")
    cfg_infer = CleanConfig(
        label_col="label",
        day_col="day",
        min_non_null_frac=0.40,
        debug=False,
        mode="infer",
        use_ts_features=True,
    )

    # Se x_test è separato da ';' e con colonna "Unnamed: 0", usa read_user_csv del tuo preprocessing.
    # Se invece è veramente ';' con problemi di parsing, usa load_semicolon_csv_keep_rows.
    x_test = read_user_csv(str(test_path))
    x_test = clean_user_df(x_test, cfg_infer)

    # id output
    if "id" in x_test.columns:
        ids = pd.to_numeric(x_test["id"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        ids = np.arange(len(x_test), dtype=int)

    # features
    X = x_test.copy()
    X = X.drop(columns=[c for c in ["id", "label", "date"] if c in X.columns], errors="ignore")
    X = clean_dataframe_soft(X)

    # allinea alle train_features
    for c in train_features:
        if c not in X.columns:
            X[c] = 0
    X = X[train_features].replace([np.inf, -np.inf], np.nan).fillna(0)

    # predict ensemble
    y_pred = predict_ensemble(ensemble_items, X.copy())
    if y_pred.size == 0:
        print("❌ Errore: predizioni vuote su x_test.csv")
        return

    y_pred = np.rint(y_pred).astype(int)
    out = pd.DataFrame({"id": ids, "label": y_pred})
    out.to_csv(BASE_DIR / "predictions.csv", index=False)
    print("✅ Creato predictions.csv")


if __name__ == "__main__":
    main()
