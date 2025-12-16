import json
import pickle
import io
import numpy as np
import pandas as pd
import flwr as fl
from sklearn.decomposition import non_negative_factorization
from sklearn.metrics import mean_absolute_error

from config import SERVER_ADDRESS, NUM_ROUNDS
from strategy import RandomForestAggregation


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
    """Pulizia soft: NON elimina righe."""
    # drop colonne tipo Unnamed: 0
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

    # strip nomi colonna e rimuovi duplicati
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    # inf -> NaN (poi fillna)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def main():
    strategy = RandomForestAggregation(
        top_k=50,
        save_path="selected_features.json",
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
    with open("global_model.pkl", "rb") as f:
        model = pickle.load(f)

    # -------------------------
    # GLOBAL TEST su HOLDOUT client
    # -------------------------
    HOLDOUT_CID = 8
    if HOLDOUT_CID > 8:
        holdout_path = f"../clients_data/group{HOLDOUT_CID}_merged_clean.csv"

        # NB: lascio come nel tuo codice (dropna) per non cambiare la tua metrica holdout
        holdout = pd.read_csv(holdout_path, sep=",").dropna()

        # target
        y_holdout = holdout["label"].copy()

        # droppa meta-colonne come nei client
        cols_to_drop = ["day", "client_id", "user_id", "source_file", "label"]
        cols_to_drop = [c for c in cols_to_drop if c in holdout.columns]
        X_holdout = holdout.drop(columns=cols_to_drop)

        # allinea feature al modello globale
        with open("global_model_features.json", "r", encoding="utf-8") as f:
            train_features = json.load(f)["features"]

        for c in train_features:
            if c not in X_holdout.columns:
                X_holdout[c] = 0

        X_holdout = X_holdout[train_features].fillna(0)

        y_pred_holdout = model.predict(X_holdout)
        mae_holdout = mean_absolute_error(y_holdout, y_pred_holdout)
        print(f"✅ GLOBAL HOLDOUT MAE (client {HOLDOUT_CID}): {mae_holdout:.4f}")

    # -------------------------
    # TEST FINALE SU x_test.csv
    # -------------------------

    # 2) carica x_test.csv (robusto: non perde righe anche se alcune sono 'rotte')
    x_test = load_semicolon_csv_keep_rows("x_test.csv")
    x_test = clean_dataframe_soft(x_test)

    # id: se non esiste, lo creo
    if "id" in x_test.columns:
        ids = pd.to_numeric(x_test["id"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        ids = np.arange(len(x_test), dtype=int)

    # costruisco X
    X = x_test.copy()

    # togli colonne che sicuramente non sono feature (solo colonne, mai righe)
    X = X.drop(columns=[c for c in ["id", "label", "date"] if c in X.columns], errors="ignore")
    X = clean_dataframe_soft(X)

    # carico le feature del training (salvate dalla strategy)
    with open("global_model_features.json", "r", encoding="utf-8") as f:
        train_features = json.load(f)["features"]

    # 1) aggiungi eventuali colonne mancanti (NaN, poi fillna)
    for c in train_features:
        if c not in X.columns:
            X[c] = np.nan

    # 2) droppa colonne extra e riordina
    X = X[train_features]

    # 3) pulizia NaN (senza eliminare righe)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # predizione
    y_pred = model.predict(X)
    y_pred = np.rint(y_pred).astype(int)

    out = pd.DataFrame({"id": ids, "label": y_pred})
    out.to_csv("predictions.csv", index=False)
    print("✅ Creato predictions.csv (formato: id,label)")


if __name__ == "__main__":
    main()
