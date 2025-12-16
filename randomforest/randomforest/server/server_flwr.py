import json
import pickle
import numpy as np
import pandas as pd
import flwr as fl
from sklearn.metrics import mean_absolute_error

from config import SERVER_ADDRESS, NUM_ROUNDS
from strategy import RandomForestAggregation


def main():
    strategy = RandomForestAggregation(
        top_k=15,
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
    holdout_path = f"../clients_data/group{HOLDOUT_CID}_merged_clean.csv"

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


    # 2) carica x_test.csv
    x_test = pd.read_csv("x_test.csv", sep=";")  # cambia sep se serve

    # id: se non esiste, lo creo
    if "id" in x_test.columns:
        ids = x_test["id"].astype(int).to_numpy()
    else:
        ids = np.arange(len(x_test), dtype=int)

    # costruisco X
    X = x_test.copy()

    # togli colonne che sicuramente non sono feature
    X = X.drop(columns=[c for c in ["id", "label", "date"] if c in X.columns], errors="ignore")

    # carico le feature del training (salvate dalla strategy)
    with open("global_model_features.json", "r", encoding="utf-8") as f:
        train_features = json.load(f)["features"]

    # 1) aggiungi eventuali colonne mancanti (metto 0)
    for c in train_features:
        if c not in X.columns:
            X[c] = 0

    # 2) droppa colonne extra (tipo *_time_series, act_activeTime, ecc.)
    X = X[train_features]

    # 3) pulizia NaN (se presenti)
    X = X.fillna(0)

    # predizione
    y_pred = model.predict(X)
    y_pred = np.rint(y_pred).astype(int)

    out = pd.DataFrame({"id": ids, "label": y_pred})
    out.to_csv("predictions.csv", index=False)
    print("✅ Creato predictions.csv (formato: id,label)")


if __name__ == "__main__":
    main()
