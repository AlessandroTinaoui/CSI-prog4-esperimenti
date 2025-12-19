import io
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from config import HOLDOUT_CID, NUM_ROUNDS, SERVER_ADDRESS, LOCAL_BOOST_ROUND, TOP_K_FEATURES
from strategy import XGBoostTreeAppendStrategy

# -------------------------------------------------------------------
# Config + Strategy
# -------------------------------------------------------------------
from config import HOLDOUT_CID, NUM_ROUNDS, SERVER_ADDRESS  # noqa: E402
from strategy import XGBoostEnsembleAggregation  # noqa: E402


# -------------------------
# Utils
# -------------------------
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
    """Pulizia 'soft': non droppa righe, solo sistemazioni colonne/inf."""
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


def main():
    base_dir = Path(__file__).resolve().parent
    global_model_path = base_dir / "global_model.json"
    global_features_path = base_dir / "global_model_features.json"

    for p in [global_model_path, global_features_path]:
        if p.exists():
            p.unlink()

    strategy = XGBoostTreeAppendStrategy(
        top_k=TOP_K_FEATURES,
        save_path="selected_features.json",
        local_boost_round=LOCAL_BOOST_ROUND,
        global_model_path="global_model.json",
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
        print("❌ ERRORE: file global_model.json / global_model_features.json non creati.")
        sys.exit(1)

    booster = _load_global_booster(global_model_path)
    print("✅ Modello globale caricato.")

    train_features = json.loads(global_features_path.read_text(encoding="utf-8"))["features"]

    # -------------------------
    # GLOBAL TEST su HOLDOUT
    # -------------------------
    if HOLDOUT_CID <= 8:
        holdout_path = base_dir / "../clients_data" / f"group{HOLDOUT_CID}_merged_clean.csv"
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

            model_feats = booster.feature_names
            if model_feats:
                for c in model_feats:
                    if c not in X_holdout.columns:
                        X_holdout[c] = 0
                X_holdout = X_holdout[model_feats]

            dmat = xgb.DMatrix(X_holdout, feature_names=model_feats or train_features)
            y_pred_holdout = booster.predict(dmat)
            mae_holdout = mean_absolute_error(y_holdout, y_pred_holdout)
            print(f"📊 GLOBAL HOLDOUT MAE (client {HOLDOUT_CID}): {mae_holdout:.4f}")
        else:
            print(f"⚠️ Holdout non trovato: {holdout_path}")

    # -------------------------
    # TEST FINALE SU x_test.csv
    # -------------------------
    test_path = base_dir / "x_test.csv"
    if not test_path.exists():
        print(f"⚠️ File x_test.csv non trovato in {test_path}")
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
            X[c] = np.nan
    X = X[train_features]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    model_feats = booster.feature_names
    if model_feats:
        for c in model_feats:
            if c not in X.columns:
                X[c] = 0
        X = X[model_feats]

    dmat = xgb.DMatrix(X, feature_names=model_feats or train_features)
    y_pred = booster.predict(dmat)

    y_pred_int = np.rint(y_pred).astype(int)
    out = pd.DataFrame({"id": ids, "label": y_pred_int})
    out.to_csv("predictions.csv", index=False)
    print("✅ Creato predictions.csv")


if __name__ == "__main__":
    main()
