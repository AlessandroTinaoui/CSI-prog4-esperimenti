import json
import pickle
import numpy as np
import pandas as pd
import flwr as fl

from config import SERVER_ADDRESS, NUM_ROUNDS
from strategy import RandomForestAggregation


def main():
    strategy = RandomForestAggregation(
        top_k=15,
        save_path="selected_features.json",
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=9,
        min_evaluate_clients=9,
        min_available_clients=9,
    )

    # Avvia FL (bloccante). Quando finisce, continuiamo con il test finale.
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    # -------------------------
    # TEST FINALE SU x_test.csv
    # -------------------------
    # 1) carica modello globale salvato dalla strategy
    with open("global_model.pkl", "rb") as f:
        model = pickle.load(f)

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
