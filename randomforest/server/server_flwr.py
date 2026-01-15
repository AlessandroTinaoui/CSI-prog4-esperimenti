import json
import os
import pickle
import io
from pathlib import Path

import numpy as np
import pandas as pd
import flwr as fl
from sklearn.metrics import mean_absolute_error

from config import SERVER_ADDRESS, NUM_ROUNDS, HOLDOUT_CID
from strategy import RandomForestAggregation
from dataset.dataset_cfg import get_train_path, get_test_path


def load_semicolon_csv_keep_rows(path: str) -> pd.DataFrame:
    """
    Carica un CSV separato da ';' senza perdere righe:
    - usa la prima riga come header
    - per ogni riga: pad se mancano campi, accorpa se ce ne sono troppi
    (utile se alcune righe sono 'rotte' e sballano il numero colonne)
    """
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
            # accorpa l'eccesso nell'ultima colonna per non perdere dati
            parts = parts[: ncol - 1] + [";".join(parts[ncol - 1 :])]

        fixed_lines.append(";".join(parts))

    text = "\n".join(fixed_lines)
    return pd.read_csv(io.StringIO(text), sep=";", engine="python")


def clean_dataframe_soft(df: pd.DataFrame) -> pd.DataFrame:
    # drop colonne tipo Unnamed: 0
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

    # strip nomi colonna e rimuovi duplicati
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    df = df.replace([np.inf, -np.inf], np.nan)
    return df

WORKDIR = Path.cwd()

RESULTS_DIR = (WORKDIR / ".." / "results").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    TRAIN_PATH = get_train_path()
    TEST_PATH = get_test_path()
    selected_path = str(RESULTS_DIR / "selected_features.json")
    print(str(selected_path))
    strategy = RandomForestAggregation(
        top_k=50,
        save_path=selected_path,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=8,
    )

    # Avvia FL (bloccante). Quando finisce, continuiamo con il test finale.
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    # 1) carica modello globale salvato dalla strategy
    with open("../results/global_model.pkl", "rb") as f:
        model = pickle.load(f)

    # -------------------------
    # GLOBAL TEST su HOLDOUT client
    # -------------------------
    if HOLDOUT_CID <= 8:
        holdout_path = "../../"+TRAIN_PATH+"/group"+str(HOLDOUT_CID)+"_merged_clean.csv"
        print(holdout_path)
        # NB: lascio come nel tuo codice (dropna) per non cambiare la tua metrica holdout
        holdout = pd.read_csv(holdout_path, sep=",").dropna()

        # target
        y_holdout = holdout["label"].copy()

        # droppa meta-colonne come nei client
        cols_to_drop = ["day", "client_id", "user_id", "source_file", "label"]
        cols_to_drop = [c for c in cols_to_drop if c in holdout.columns]
        X_holdout = holdout.drop(columns=cols_to_drop)

        # allinea feature al modello globale
        with open("../results/global_model_features.json", "r", encoding="utf-8") as f:
            train_features = json.load(f)["features"]

        for c in train_features:
            if c not in X_holdout.columns:
                X_holdout[c] = 0

        X_holdout = X_holdout[train_features].fillna(0)

        y_pred_holdout = model.predict(X_holdout)
        mae_holdout = mean_absolute_error(y_holdout, y_pred_holdout)
        print(f"MEA valutato sul client {HOLDOUT_CID}")
        print(f"FINAL_MAE: {mae_holdout}")

    # -------------------------
    # TEST FINALE SU x_test.csv
    # -------------------------

    x_test = pd.read_csv("../../" + TEST_PATH, sep=",")
    x_test = x_test.drop(columns=["Unnamed: 0"], errors="ignore")
    x_test.columns = x_test.columns.astype(str).str.strip()

    # id
    if "id" in x_test.columns:
        ids = pd.to_numeric(x_test["id"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        ids = np.arange(len(x_test), dtype=int)

    # base X: tutte le colonne tranne quelle non-feature
    X = x_test.drop(columns=[c for c in ["id", "label", "date"] if c in x_test.columns], errors="ignore")

    # prova a caricare le feature del training (se ci sono)
    train_features = None
    feat_path = "../results/global_model_features.json"

    if os.path.exists(feat_path):
        try:
            with open(feat_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            feats = data.get("features", None)
            if isinstance(feats, list):
                feats = [str(c).strip() for c in feats if str(c).strip()]
                if len(feats) > 0:
                    train_features = feats
        except Exception as e:
            print(f"[WARN] Non riesco a leggere {feat_path}: {e}. Uso tutte le colonne di x_test.")

    if train_features is not None:
        X = X.reindex(columns=train_features)  # aggiunge mancanti come NaN, elimina extra, riordina
    else:
        print("[INFO] train_features assenti/vuote -> uso tutte le colonne di x_test (dopo drop id/label/date).")

    X = X.apply(pd.to_numeric, errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)



    # predizione
    y_pred = model.predict(X)
    y_pred = np.rint(y_pred).astype(int)

    out = pd.DataFrame({"id": ids, "label": y_pred})
    out.to_csv("../results/predictions.csv", index=False)
    print("Creato predictions.csv (formato: id,label)")


if __name__ == "__main__":
    main()
