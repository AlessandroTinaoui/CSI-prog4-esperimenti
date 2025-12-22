# nnmodel/client/client_app.py

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import flwr as fl
from flwr.client import NumPyClient
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nnmodel.model import MLPRegressor
from nnmodel.data import (
    load_csv_dataset,
    split_train_test,
    ensure_feature_order_and_fill,
    apply_standardization,
    local_sums_for_scaler,
    ScalerStats,
)

# -------- Paths/logs (stile simile al tuo) --------
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

def train_one_client(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

def eval_one_client(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    clip_min: float,
    clip_max: float,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        pred = model(xb).detach().cpu().numpy()
    pred = np.clip(pred, clip_min, clip_max)
    return float(mean_absolute_error(y, pred))


class NNClient(NumPyClient):
    def __init__(self, cid: int, data_path: str, cfg: Dict):
        self.cid = int(cid)
        self.data_path = data_path
        self.cfg = cfg

        self.logger = logging.getLogger(f"client_{self.cid}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(LOGS_DIR / f"client_{self.cid}_log.txt")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Load raw (preprocessed) CSV
        X_df, y_sr = load_csv_dataset(data_path, label_col="label")
        self.feature_names_local = list(X_df.columns)

        X_tr_df, X_te_df, y_tr, y_te = split_train_test(
            X_df, y_sr,
            test_size=self.cfg["TEST_SIZE"],
            random_state=self.cfg["RANDOM_STATE"],
            shuffle=self.cfg["SHUFFLE_SPLIT"],
        )

        self.X_train_df = X_tr_df
        self.X_test_df = X_te_df
        self.y_train = y_tr.to_numpy(dtype=np.float32)
        self.y_test = y_te.to_numpy(dtype=np.float32)

        self.global_features: Optional[List[str]] = None
        self.scaler: Optional[ScalerStats] = None

        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Loaded {data_path} | X_train={self.X_train_df.shape}, X_test={self.X_test_df.shape}")

    def _build_model_if_needed(self, input_dim: int):
        if self.model is None:
            self.model = MLPRegressor(
                input_dim=input_dim,
                hidden_sizes=self.cfg["HIDDEN_SIZES"],
                dropout=self.cfg["DROPOUT"],
            ).to(self.device)

    # Flower required
    def get_parameters(self, config):
        if self.model is None:
            # temporaneo: model verrà creato quando conosciamo input_dim (dopo features globali)
            return []
        return get_model_params(self.model)

    def fit(self, parameters, config):
        phase = (config or {}).get("phase", "train")

        # -------------------------
        # ROUND 1: SCALER STATS
        # -------------------------
        if phase == "scaler":
            # invia feature_names + n, sum, sumsq su TRAIN
            feat_names = self.feature_names_local
            n, s, ssq = local_sums_for_scaler(self.X_train_df.fillna(0.0), feat_names)

            metrics = {
                "feature_names": json.dumps(feat_names),
                "n": int(n),
                "sum": json.dumps(s.tolist()),
                "sumsq": json.dumps(ssq.tolist()),
            }
            self.logger.info("Phase=scaler: sent n/sum/sumsq")
            return [], n, metrics

        # -------------------------
        # ROUND >=2: TRAIN
        # -------------------------
        # Ricevi feature list + scaler dal server
        gf = json.loads(config["global_features"])
        mean = np.array(json.loads(config["scaler_mean"]), dtype=np.float32)
        std = np.array(json.loads(config["scaler_std"]), dtype=np.float32)

        self.global_features = list(gf)
        self.scaler = ScalerStats(mean=mean, std=std)

        # Allinea + standardizza
        Xtr = ensure_feature_order_and_fill(self.X_train_df.copy(), self.global_features)
        Xte = ensure_feature_order_and_fill(self.X_test_df.copy(), self.global_features)

        Xtr = apply_standardization(Xtr, self.scaler)
        Xte = apply_standardization(Xte, self.scaler)

        # Build model (input_dim noto ora)
        self._build_model_if_needed(input_dim=Xtr.shape[1])

        # Set global params
        if parameters and len(parameters) > 0:
            set_model_params(self.model, parameters)

        # Dataloader
        ds = TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=self.cfg["BATCH_SIZE"], shuffle=True)

        train_one_client(
            self.model, loader,
            epochs=self.cfg["LOCAL_EPOCHS"],
            lr=self.cfg["LR"],
            weight_decay=self.cfg["WEIGHT_DECAY"],
            device=self.device,
        )

        # Train MAE (monitor)
        mae_train = eval_one_client(
            self.model, Xtr, self.y_train,
            clip_min=self.cfg["CLIP_MIN"], clip_max=self.cfg["CLIP_MAX"],
            device=self.device,
        )

        metrics = {"train_mae": float(mae_train)}
        return get_model_params(self.model), len(Xtr), metrics

    def evaluate(self, parameters, config):
        if self.global_features is None or self.scaler is None:
            # se evaluate viene chiamato prima del train vero
            return float("nan"), 0, {"eval_mae": float("nan")}

        # Set params
        if parameters and len(parameters) > 0:
            set_model_params(self.model, parameters)

        Xte = ensure_feature_order_and_fill(self.X_test_df.copy(), self.global_features)
        Xte = apply_standardization(Xte, self.scaler)

        mae = eval_one_client(
            self.model, Xte, self.y_test,
            clip_min=self.cfg["CLIP_MIN"], clip_max=self.cfg["CLIP_MAX"],
            device=self.device,
        )
        return float(mae), len(Xte), {"eval_mae": float(mae)}


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

    # Config “client-side” (stessa per tutti)
    from nnmodel.server.config import (
        TEST_SIZE, RANDOM_STATE, SHUFFLE_SPLIT,
        LOCAL_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
        HIDDEN_SIZES, DROPOUT,
        CLIP_MIN, CLIP_MAX
    )

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
    }

    client = NNClient(cid=client_id, data_path=data_path, cfg=cfg)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
