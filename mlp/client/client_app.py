# mlp/client/client_app.py

import os
import sys
import json
import logging
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.client import NumPyClient
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mlp.model import MLPRegressor
from mlp.data import (
    load_csv_dataset,
    split_train_test,
    ensure_feature_order_and_fill,
    apply_standardization,
    local_sums_for_scaler,
    ScalerStats,
)
from mlp.server.config import SERVER_ADDRESS

# Import parametri (incluso early stopping)
from client_params import (
    BETA,
    ES_ENABLED,
    ES_VAL_FRAC,
    ES_PATIENCE,
    ES_MIN_DELTA,
    ES_MIN_EPOCHS,
    ES_RESTORE_BEST,
)

# -------- Paths/logs --------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CLIENTS_DATA_DIR = Path(__file__).resolve().parent / "clients_data"


def get_model_params(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_params(model: nn.Module, params: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(params):
        raise ValueError(f"Param mismatch: got {len(params)} arrays, expected {len(keys)}")
    new_state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(new_state, strict=True)


def eval_one_client_real_mae(
    model: nn.Module,
    X: np.ndarray,
    y_real: np.ndarray,
    clip_min: float,
    clip_max: float,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> float:
    """Valuta MAE in scala reale (0..100) anche se il modello predice y normalizzata."""
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        pred_norm = model(xb).detach().cpu().numpy()

    pred = pred_norm * y_std + y_mean
    pred = np.clip(pred, clip_min, clip_max)
    return float(mean_absolute_error(y_real, pred))


def _time_aware_train_val_split(
    X: np.ndarray,
    y_norm: np.ndarray,
    y_real: np.ndarray,
    val_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split time-aware: validation = ultimo blocco del train (niente shuffle)."""
    n = int(X.shape[0])
    if n < 5:
        # troppo piccolo: niente val
        return X, y_norm, y_real, X[:0], y_norm[:0], y_real[:0]

    val_n = max(1, int(round(n * val_frac)))
    val_n = min(val_n, n - 1)  # lascia almeno 1 elemento al train

    split = n - val_n
    X_tr, X_val = X[:split], X[split:]
    y_trn, y_valn = y_norm[:split], y_norm[split:]
    y_trr, y_valr = y_real[:split], y_real[split:]
    return X_tr, y_trn, y_trr, X_val, y_valn, y_valr


def train_one_client_with_early_stopping(
    model: nn.Module,
    Xtr: np.ndarray,
    ytr_norm: np.ndarray,
    ytr_real: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    clip_min: float,
    clip_max: float,
    y_mean: float,
    y_std: float,
    es_enabled: bool,
    es_val_frac: float,
    es_patience: int,
    es_min_delta: float,
    es_min_epochs: int,
    es_restore_best: bool,
) -> Dict[str, float]:
    """
    Training locale con early stopping su MAE in scala reale calcolata su validation interno (time-aware).
    Restituisce metriche utili (best_val_mae, epochs_ran).
    """
    model.to(device)

    # split train/val interno
    X_train, y_train_norm, y_train_real, X_val, y_val_norm, y_val_real = _time_aware_train_val_split(
        Xtr, ytr_norm, ytr_real, es_val_frac
    )

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_norm, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=BETA)

    # se val vuoto (dataset troppo piccolo), fallback: training classico
    if (not es_enabled) or (X_val.shape[0] == 0):
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        return {"best_val_mae": float("nan"), "epochs_ran": float(epochs)}

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        # ---- train 1 epoch
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        # ---- eval on val (MAE reale)
        val_mae = eval_one_client_real_mae(
            model=model,
            X=X_val,
            y_real=y_val_real,
            clip_min=clip_min,
            clip_max=clip_max,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
        )

        improved = (best_val - val_mae) > es_min_delta
        if improved:
            best_val = val_mae
            bad_epochs = 0
            if es_restore_best:
                best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        # early stop solo dopo un minimo di epoche
        if epoch >= es_min_epochs and bad_epochs >= es_patience:
            break

    if es_restore_best and best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return {"best_val_mae": float(best_val), "epochs_ran": float(epoch)}


class NNClient(NumPyClient):
    def __init__(self, cid: int, data_path: str, cfg: Dict):
        self.cid = int(cid)
        self.data_path = data_path
        self.cfg = cfg

        self.logger = logging.getLogger(f"client_{self.cid}")
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(LOGS_DIR / f"client_{self.cid}.log")
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load local data
        X_df, y = load_csv_dataset(self.data_path, label_col="label")

        # Split local train/test (coerente con la tua pipeline)
        X_train_df, X_test_df, y_train, y_test = split_train_test(
            X_df,
            y,
            test_size=self.cfg["TEST_SIZE"],
            random_state=self.cfg["RANDOM_STATE"],
            shuffle=self.cfg["SHUFFLE_SPLIT"],
        )

        self.X_train_df = X_train_df
        self.X_test_df = X_test_df
        self.y_train = y_train.to_numpy(dtype=np.float32)
        self.y_test = y_test.to_numpy(dtype=np.float32)

        self.feature_names_local = list(self.X_train_df.columns)

        self.global_features: Optional[List[str]] = None
        self.scaler: Optional[ScalerStats] = None
        self.model: Optional[nn.Module] = None

        self.logger.info(f"Client {self.cid} ready. Train={len(self.X_train_df)}, Test={len(self.X_test_df)}")

    def _build_model_if_needed(self, input_dim: int) -> None:
        if self.model is None:
            self.model = MLPRegressor(
                input_dim=input_dim,
                hidden_sizes=self.cfg["HIDDEN_SIZES"],
                dropout=self.cfg["DROPOUT"],
            ).to(self.device)
            self.logger.info(f"Built MLPRegressor input_dim={input_dim}, hidden={self.cfg['HIDDEN_SIZES']}")

    def get_parameters(self, config):
        if self.model is None:
            return []
        return get_model_params(self.model)

    def fit(self, parameters, config):
        phase = (config or {}).get("phase", "train")

        # -------------------------
        # ROUND 1: SCALER STATS (X) + STATS TARGET (Y)
        # -------------------------
        if phase == "scaler":
            feat_names = self.feature_names_local
            n, s, ssq = local_sums_for_scaler(self.X_train_df.fillna(0.0), feat_names)

            # stats y (solo somme)
            ytr64 = self.y_train.astype(np.float64)
            y_n = int(ytr64.shape[0])
            y_sum = float(np.sum(ytr64))
            y_sumsq = float(np.sum(ytr64 * ytr64))

            metrics = {
                "feature_names": json.dumps(feat_names),
                "n": int(n),
                "sum": json.dumps(s.tolist()),
                "sumsq": json.dumps(ssq.tolist()),
                "y_n": y_n,
                "y_sum": y_sum,
                "y_sumsq": y_sumsq,
            }
            self.logger.info("Phase=scaler: sent X n/sum/sumsq + Y n/sum/sumsq")
            return [], n, metrics

        # -------------------------
        # ROUND >=2: TRAIN
        # -------------------------
        gf = json.loads(config["global_features"])
        mean = np.array(json.loads(config["scaler_mean"]), dtype=np.float32)
        std = np.array(json.loads(config["scaler_std"]), dtype=np.float32)

        # y global mean/std
        y_mean = float(config.get("y_mean", 0.0))
        y_std = float(config.get("y_std", 1.0))
        if y_std <= 1e-12:
            y_std = 1.0

        self.global_features = list(gf)
        self.scaler = ScalerStats(mean=mean, std=std)

        # Allinea + standardizza X
        Xtr = ensure_feature_order_and_fill(self.X_train_df.copy(), self.global_features)
        Xtr = apply_standardization(Xtr, self.scaler)

        # Build model
        self._build_model_if_needed(input_dim=Xtr.shape[1])

        # Set global params
        if parameters and len(parameters) > 0:
            set_model_params(self.model, parameters)

        # y normalizzata (per loss)
        ytr_norm = (self.y_train - y_mean) / y_std

        # ---- TRAIN con early stopping locale
        es_metrics = train_one_client_with_early_stopping(
            model=self.model,
            Xtr=Xtr,
            ytr_norm=ytr_norm.astype(np.float32),
            ytr_real=self.y_train.astype(np.float32),
            epochs=self.cfg["LOCAL_EPOCHS"],
            batch_size=self.cfg["BATCH_SIZE"],
            lr=self.cfg["LR"],
            weight_decay=self.cfg["WEIGHT_DECAY"],
            device=self.device,
            clip_min=self.cfg["CLIP_MIN"],
            clip_max=self.cfg["CLIP_MAX"],
            y_mean=y_mean,
            y_std=y_std,
            es_enabled=self.cfg["ES_ENABLED"],
            es_val_frac=self.cfg["ES_VAL_FRAC"],
            es_patience=self.cfg["ES_PATIENCE"],
            es_min_delta=self.cfg["ES_MIN_DELTA"],
            es_min_epochs=self.cfg["ES_MIN_EPOCHS"],
            es_restore_best=self.cfg["ES_RESTORE_BEST"],
        )

        # Monitor MAE train (reale)
        mae_train = eval_one_client_real_mae(
            self.model,
            Xtr,
            self.y_train,
            clip_min=self.cfg["CLIP_MIN"],
            clip_max=self.cfg["CLIP_MAX"],
            device=self.device,
            y_mean=y_mean,
            y_std=y_std,
        )

        metrics = {
            "train_mae_real": float(mae_train),
            "es_best_val_mae_real": float(es_metrics["best_val_mae"]),
            "es_epochs_ran": float(es_metrics["epochs_ran"]),
        }
        return get_model_params(self.model), len(Xtr), metrics

    def evaluate(self, parameters, config):
        if self.global_features is None or self.scaler is None or self.model is None:
            return float("nan"), 0, {"eval_mae_real": float("nan")}

        if parameters and len(parameters) > 0:
            set_model_params(self.model, parameters)

        # y global mean/std
        y_mean = float(config.get("y_mean", 0.0))
        y_std = float(config.get("y_std", 1.0))
        if y_std <= 1e-12:
            y_std = 1.0

        Xte = ensure_feature_order_and_fill(self.X_test_df.copy(), self.global_features)
        Xte = apply_standardization(Xte, self.scaler)

        mae = eval_one_client_real_mae(
            self.model,
            Xte,
            self.y_test,
            clip_min=self.cfg["CLIP_MIN"],
            clip_max=self.cfg["CLIP_MAX"],
            device=self.device,
            y_mean=y_mean,
            y_std=y_std,
        )
        return float(mae), len(Xte), {"eval_mae_real": float(mae)}


if __name__ == "__main__":
    # Usage: python client_app.py <client_id> [optional_csv_path]
    if len(sys.argv) < 2:
        print("Usage: python client_app.py <client_id> [optional_csv_path]")
        sys.exit(1)

    client_id = int(sys.argv[1])
    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    else:
        data_path = str(CLIENTS_DATA_DIR / f"group{client_id}_merged_clean.csv")

    if not os.path.exists(data_path):
        print(f"ERRORE: Data file not found: {data_path}")
        sys.exit(1)

    # Importa tutto dal file parametri (come facevi già)
    from client_params import *  # noqa: F403,F401

    cfg = {
        "TEST_SIZE": TEST_SIZE,
        "RANDOM_STATE": RANDOM_STATE,
        "SHUFFLE_SPLIT": SHUFFLE_SPLIT,
        "LOCAL_EPOCHS": LOCAL_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "HIDDEN_SIZES": HIDDEN_SIZES,
        "DROPOUT": DROPOUT,
        "CLIP_MIN": CLIP_MIN,
        "CLIP_MAX": CLIP_MAX,

        # early stopping
        "ES_ENABLED": ES_ENABLED,
        "ES_VAL_FRAC": ES_VAL_FRAC,
        "ES_PATIENCE": ES_PATIENCE,
        "ES_MIN_DELTA": ES_MIN_DELTA,
        "ES_MIN_EPOCHS": ES_MIN_EPOCHS,
        "ES_RESTORE_BEST": ES_RESTORE_BEST,
    }

    client = NNClient(cid=client_id, data_path=data_path, cfg=cfg)
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)
