from __future__ import annotations

import os
import glob
import numpy as np

from config import DataConfig, FeatureSelectionConfig, RFLocalConfig, FederatedConfig
from data_io import load_client_merged_csv
from Progetto_0_nonflwr.feature_selection import get_feature_columns, aggregate_importances_weighted, select_top_k
from Progetto_0_nonflwr.rf_federated import build_global_forest, eval_regression
from Progetto_0_nonflwr.client import FederatedClient


def run_federated(
    client_dir: str,
    dc: DataConfig,
    fs_cfg: FeatureSelectionConfig,
    rf_cfg: RFLocalConfig,
    fed_cfg: FederatedConfig,
):
    pattern = os.path.join(client_dir, "group*_merged_clean.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Nessun file client trovato: {pattern}")

    # Carico e istanzio client
    clients = []
    for p in paths:
        client_id = os.path.basename(p).split("_merged_clean.csv")[0]  # group0...
        df = load_client_merged_csv(p)
        clients.append(FederatedClient(client_id, df, dc))

    # Features (uguali per ipotesi)
    feats = get_feature_columns(clients[0].df, dc)
    print(f"[INFO] Features totali: {len(feats)}")

    # -------------------------
    # ROUND 0: Feature Selection federata
    # -------------------------
    payloads = []
    for c in clients:
        r = c.compute_importances(rf_cfg, fs_cfg, feats)
        payloads.append((r.n_samples, r.importances))

    global_imps = aggregate_importances_weighted(payloads)
    selected = select_top_k(global_imps, fs_cfg.k_top)

    print(f"[INFO] Top-{fs_cfg.k_top} features selezionate:")
    print(selected)

    # -------------------------
    # ROUNDS: training federato (naive = concatenazione alberi)
    # -------------------------
    global_estimators = []

    for rnd in range(1, fed_cfg.rounds + 1):
        round_estimators = []
        for c in clients:
            tr = c.train_local(rf_cfg, selected)
            round_estimators.extend(tr.estimators)

        global_estimators.extend(round_estimators)
        if len(global_estimators) > fed_cfg.max_global_trees:
            global_estimators = global_estimators[-fed_cfg.max_global_trees:]

        global_model = build_global_forest(
            rf_cfg,
            global_estimators,
            n_features=len(selected),
            feature_names=selected
        )

        # Eval pesata
        rmses, maes, weights = [], [], []
        for c in clients:
            m = eval_regression(global_model, c.df, dc, selected)
            rmses.append(m["rmse"])
            maes.append(m["mae"])
            weights.append(len(c.df))

        w = np.array(weights, dtype=float)
        w = w / w.sum()

        print(
            f"[ROUND {rnd}] trees={len(global_estimators)} | "
            f"wRMSE={float(np.sum(w*np.array(rmses))):.4f} | "
            f"wMAE={float(np.sum(w*np.array(maes))):.4f}"
        )

    return selected


if __name__ == "__main__":
    # Se esegui server.py direttamente, punta alla tua cartella dei merged clean
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CLIENT_DIR = os.path.join(SCRIPT_DIR, "..", "dataset", "CSV_train", "CSV_train_clean")  # ADATTA SE SERVE

    dc = DataConfig(label_col="label", day_col="day")
    fs_cfg = FeatureSelectionConfig(k_top=30, scoring="neg_root_mean_squared_error", n_repeats=5)
    rf_cfg = RFLocalConfig(n_estimators=80)
    fed_cfg = FederatedConfig(rounds=5, max_global_trees=400)

    run_federated(CLIENT_DIR, dc, fs_cfg, rf_cfg, fed_cfg)
