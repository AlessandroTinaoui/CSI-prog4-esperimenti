from __future__ import annotations

import argparse
import time

import flwr as fl

from typing import Dict, List, Tuple, Optional

from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays

from common_flwr import (
    loads_json_uint8, dumps_to_uint8, loads_from_uint8
)


class RFNaiveBaggingStrategy(fl.server.strategy.Strategy):
    """
    Round 1: Feature selection (clients send {importances})
    Round >=2: Training (clients send pickled estimators)
    Aggregation: concat estimators, cap to max_global_trees
    """

    def __init__(self, num_clients: int, k_top: int, rounds: int, max_global_trees: int, seed: int = 42):
        self.num_clients = num_clients
        self.k_top = k_top
        self.rounds = rounds
        self.max_global_trees = max_global_trees
        self.seed = seed

        self.selected_feats: Optional[List[str]] = None
        self.global_estimators: List = []

        # FS config
        self.scoring = "neg_root_mean_squared_error"
        self.n_repeats = 5
        self.val_size = 0.2
        self.rf_fs_estimators = 80
        self.rf_train_estimators = 80

    def _wait_for_clients(self, client_manager, min_clients: int):
        while client_manager.num_available() < min_clients:
            print(f"[WAIT] clients available={client_manager.num_available()} < {min_clients} ...", flush=True)
            time.sleep(1.0)

    # ---- required Strategy interface ----
    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters([])  # no params at start

    def configure_fit(self, server_round, parameters, client_manager):
        # aspetta finché non sono connessi almeno num_clients
        self._wait_for_clients(client_manager, self.num_clients)

        clients = client_manager.sample(num_clients=self.num_clients)

        if server_round == 1:
            cfg = {
                "rnd": server_round,
                "phase": "fs",
                "scoring": self.scoring,
                "n_repeats": self.n_repeats,
                "val_size": self.val_size,
                "seed": self.seed,
                "rf_fs_estimators": self.rf_fs_estimators,
            }
            return [(c, fl.common.FitIns(parameters, cfg)) for c in clients]

        cfg = {
            "rnd": server_round,
            "phase": "train",
            "selected_feats": self.selected_feats,
            "seed": self.seed,
            "rf_train_estimators": self.rf_train_estimators,
        }
        return [(c, fl.common.FitIns(parameters, cfg)) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        if server_round == 1:
            # aggregate feature importances
            payloads: List[Tuple[int, Dict[str, float]]] = []
            feats_ref = None

            for _, fitres in results:
                arr = parameters_to_ndarrays(fitres.parameters)[0]
                payload = loads_json_uint8(arr)
                imps = payload["importances"]
                feats = payload["features"]
                feats_ref = feats if feats_ref is None else feats_ref

                n = fitres.num_examples
                payloads.append((n, imps))

            # weighted mean
            num = {f: 0.0 for f in feats_ref}
            den = 0.0
            for n, imps in payloads:
                den += n
                for f in feats_ref:
                    num[f] += n * float(imps[f])

            global_imps = {f: (num[f] / den if den else 0.0) for f in feats_ref}
            self.selected_feats = [f for f, _ in sorted(global_imps.items(), key=lambda x: x[1], reverse=True)[: self.k_top]]

            print(f"[FS] features totali={len(feats_ref)} | selezionate top-{self.k_top}={len(self.selected_feats)}")
            print(self.selected_feats)

            # nessun parametro modello da restituire ancora
            return ndarrays_to_parameters([]), {}

        # Training: concat estimators
        for _, fitres in results:
            arr = parameters_to_ndarrays(fitres.parameters)[0]
            ests = loads_from_uint8(arr)
            self.global_estimators.extend(ests)

        # cap
        if len(self.global_estimators) > self.max_global_trees:
            self.global_estimators = self.global_estimators[-self.max_global_trees :]

        # parameters = global_estimators serializzati (per evaluate)
        p = ndarrays_to_parameters([dumps_to_uint8(self.global_estimators)])
        return p, {"trees": len(self.global_estimators)}

    def configure_evaluate(self, server_round, parameters, client_manager):
        if server_round < 2 or self.selected_feats is None or len(self.global_estimators) == 0:
            return []

        self._wait_for_clients(client_manager, self.num_clients)
        clients = client_manager.sample(num_clients=self.num_clients)

        cfg = {"selected_feats": self.selected_feats, "seed": self.seed}
        return [(c, fl.common.EvaluateIns(parameters, cfg)) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        # weighted RMSE/MAE
        tot = sum(r.num_examples for _, r in results)
        w_rmse = sum(r.num_examples * r.loss for _, r in results) / tot

        # metrics are per-client; weight them too
        w_mae = 0.0
        for _, r in results:
            m = r.metrics
            w_mae += r.num_examples * float(m.get("mae", 0.0))
        w_mae /= tot

        print(f"[ROUND {server_round}] trees={len(self.global_estimators)} | wRMSE={w_rmse:.4f} | wMAE={w_mae:.4f}")
        return w_rmse, {"w_mae": w_mae}

    def evaluate(self, server_round, parameters):
        # no centralized eval
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8080")
    ap.add_argument("--num_clients", type=int, default=9)
    ap.add_argument("--rounds", type=int, default=6)  # 1 FS + (rounds-1) train
    ap.add_argument("--k_top", type=int, default=30)
    ap.add_argument("--max_trees", type=int, default=400)
    args = ap.parse_args()

    strategy = RFNaiveBaggingStrategy(
        num_clients=args.num_clients,
        k_top=args.k_top,
        rounds=args.rounds,
        max_global_trees=args.max_trees,
        seed=42,
    )

    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
