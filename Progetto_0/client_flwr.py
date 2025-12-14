from __future__ import annotations
import argparse
import flwr as fl
import numpy as np

from common_flwr import (
    load_client_csv, feature_cols,
    local_perm_importance, train_local_rf_estimators,
    build_global_model_from_estimators,
    rmse, mae,
    dumps_json_uint8, loads_json_uint8,
    dumps_to_uint8, loads_from_uint8,
)


class RFClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, csv_path: str):
        self.client_id = client_id
        self.df = load_client_csv(csv_path)
        self.all_feats = feature_cols(self.df, label_col="label", day_col="day")

        # server manderà selected_feats dopo FS
        self.selected_feats = None

    def get_parameters(self, config):
        # RF non ha "pesi" da inviare come numpy arrays in modo naturale.
        return []

    def fit(self, parameters, config):
        rnd = int(config.get("rnd", 1))
        phase = config.get("phase", "fs")  # "fs" or "train"

        if phase == "fs":
            # Feature selection round: ritorna importanze
            imps = local_perm_importance(
                self.df,
                feats=self.all_feats,
                label_col="label",
                scoring=config.get("scoring", "neg_root_mean_squared_error"),
                n_repeats=int(config.get("n_repeats", 5)),
                val_size=float(config.get("val_size", 0.2)),
                seed=int(config.get("seed", 42)),
                rf_n_estimators=int(config.get("rf_fs_estimators", 80)),
            )
            payload = {"client_id": self.client_id, "importances": imps, "features": self.all_feats}
            arr = dumps_json_uint8(payload)
            return [arr], len(self.df), {}

        # Training rounds: ricevo selected_feats dal server
        selected_feats = config.get("selected_feats", None)
        if selected_feats is None:
            raise RuntimeError("selected_feats non ricevute dal server (feature selection non completata?)")

        self.selected_feats = list(selected_feats)

        # alleno RF locale e invio alberi
        estimators = train_local_rf_estimators(
            self.df,
            feats=self.selected_feats,
            label_col="label",
            seed=int(config.get("seed", 42)),
            n_estimators=int(config.get("rf_train_estimators", 80)),
            max_depth=None,
            min_samples_leaf=1,
        )
        arr = dumps_to_uint8(estimators)
        return [arr], len(self.df), {}

    def evaluate(self, parameters, config):
        # server può inviare estimators globali per valutazione federata
        global_estimators_arr = parameters[0] if parameters else None
        if global_estimators_arr is None:
            return 0.0, len(self.df), {}

        estimators = loads_from_uint8(global_estimators_arr)
        selected_feats = config.get("selected_feats", self.selected_feats)
        if selected_feats is None:
            selected_feats = self.all_feats

        model = build_global_model_from_estimators(estimators, n_features=len(selected_feats), seed=int(config.get("seed", 42)))

        X = self.df[selected_feats]
        y = self.df["label"].astype(float)
        yhat = model.predict(X)

        metrics = {"rmse": rmse(y, yhat), "mae": mae(y, yhat)}
        # Flower vuole una loss float: usiamo RMSE come “loss”
        return metrics["rmse"], len(self.df), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_id", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--server", default="127.0.0.1:8080")
    args = ap.parse_args()

    fl.client.start_numpy_client(server_address=args.server, client=RFClient(args.client_id, args.csv))


if __name__ == "__main__":
    main()
