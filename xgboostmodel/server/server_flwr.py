import json
import pickle
import io
import os
import sys
import numpy as np
import pandas as pd
import flwr as fl
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from config import SERVER_ADDRESS, NUM_ROUNDS, HOLDOUT_CID
from strategy import XGBoostEnsembleAggregation


def load_semicolon_csv_keep_rows(path: str) -> pd.DataFrame:
    """Carica CSV robusto."""
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
        if line is None: line = ""
        parts = line.split(";")
        if len(parts) < ncol:
            parts = parts + [""] * (ncol - len(parts))
        elif len(parts) > ncol:
            parts = parts[: ncol - 1] + [";".join(parts[ncol - 1:])]
        fixed_lines.append(";".join(parts))

    text = "\n".join(fixed_lines)
    return pd.read_csv(io.StringIO(text), sep=";", engine="python")


def clean_dataframe_soft(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def main():
    # Definisce percorsi assoluti
    BASE_DIR = Path(__file__).resolve().parent
    GLOBAL_ENSEMBLE_PATH = BASE_DIR / "global_ensemble.pkl"
    GLOBAL_FEATURES_PATH = BASE_DIR / "global_model_features.json"

    # Pulisce vecchi file
    if GLOBAL_ENSEMBLE_PATH.exists(): GLOBAL_ENSEMBLE_PATH.unlink()
    if GLOBAL_FEATURES_PATH.exists(): GLOBAL_FEATURES_PATH.unlink()

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

    print("\n  FL terminato. Inizio fase di test...")

    # Verifica esistenza file prodotti
    if not GLOBAL_ENSEMBLE_PATH.exists() or not GLOBAL_FEATURES_PATH.exists():
        print("❌ ERRORE: I file del modello globale (pkl/json) non sono stati creati.")
        print("   Verifica che i client non siano crashati e che la Strategy abbia salvato i file.")
        sys.exit(1)

    # 1) Carica ensemble
    with open(GLOBAL_ENSEMBLE_PATH, "rb") as f:
        ensemble_items = pickle.load(f)
    print(f"✅ Ensemble caricato: {len(ensemble_items)} booster.")

    # 2) Carica features
    with open(GLOBAL_FEATURES_PATH, "r", encoding="utf-8") as f:
        train_features = json.load(f)["features"]

    # -------------------------
    # GLOBAL TEST su HOLDOUT
    # -------------------------
    if HOLDOUT_CID <= 8:

        # Percorso dinamico basato su posizione server
        holdout_path = "/home/alessandro/Desktop/CSI-prog4 esperimenti/xgboostmodel/clients_data/group"+str(HOLDOUT_CID)+"_merged_clean.csv"
        print(holdout_path)
        if holdout_path:
            holdout = pd.read_csv(holdout_path, sep=",").dropna()
            y_holdout = holdout["label"].copy()
            cols_to_drop = ["day", "client_id", "user_id", "source_file", "label"]
            cols_to_drop = [c for c in cols_to_drop if c in holdout.columns]
            X_holdout = holdout.drop(columns=cols_to_drop)

            # Allineamento features
            for c in train_features:
                if c not in X_holdout.columns:
                    X_holdout[c] = 0
            X_holdout = X_holdout[train_features].fillna(0)

            # Predizione (FIX: allinea feature_names al booster)
            preds = []
            weights = []
            for item in ensemble_items:
                booster = xgb.Booster()
                booster.load_model(bytearray(item["raw"]))  # bytes -> bytearray (più robusto)

                model_feats = booster.feature_names  # ordine feature del modello
                X_aligned = X_holdout[model_feats]  # riordina X in modo identico

                dmat = xgb.DMatrix(X_aligned, feature_names=model_feats)
                preds.append(booster.predict(dmat))
                weights.append(float(item.get("weight", 1.0)))

            if weights:
                weights = np.array(weights, dtype=float)
                weights /= weights.sum()
                y_pred_holdout = np.average(np.vstack(preds), axis=0, weights=weights)
                mae_holdout = mean_absolute_error(y_holdout, y_pred_holdout)
                print(f"📊 GLOBAL HOLDOUT MAE (client {HOLDOUT_CID}): {mae_holdout:.4f}")
            else:
                print(" Errore: pesi ensemble vuoti.")

    # -------------------------
    # TEST FINALE SU x_test.csv
    # -------------------------
    test_path = BASE_DIR / "x_test.csv"
    if not test_path.exists():
        print(f" File x_test.csv non trovato in {test_path}")
        return

    x_test = load_semicolon_csv_keep_rows(str(test_path))
    x_test = clean_dataframe_soft(x_test)

    if "id" in x_test.columns:
        ids = pd.to_numeric(x_test["id"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        ids = np.arange(len(x_test), dtype=int)

    X = x_test.copy()
    X = X.drop(columns=[c for c in ["id", "label", "date"] if c in X.columns], errors="ignore")
    X = clean_dataframe_soft(X)

    for c in train_features:
        if c not in X.columns:
            X[c] = np.nan

    X = X[train_features]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # FIX: allinea feature_names al booster, per evitare mismatch
    preds = []
    weights = []
    for item in ensemble_items:
        booster = xgb.Booster()
        booster.load_model(bytearray(item["raw"]))

        model_feats = booster.feature_names
        X_aligned = X[model_feats]

        dmat = xgb.DMatrix(X_aligned, feature_names=model_feats)
        preds.append(booster.predict(dmat))
        weights.append(float(item.get("weight", 1.0)))

    if weights:
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        y_pred = np.average(np.vstack(preds), axis=0, weights=weights)
        y_pred = np.rint(y_pred).astype(int)

        out = pd.DataFrame({"id": ids, "label": y_pred})
        out.to_csv("predictions.csv", index=False)
        print("Creato predictions.csv")


if __name__ == "__main__":
    main()