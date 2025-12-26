# client_app.py
from __future__ import annotations

import os
import sys
import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import flwr as fl
from flwr.client import NumPyClient
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Sampler, RandomSampler

from TabNet.model import TabNetRegressor, TabNetConfig
from TabNet.data import (
    load_csv_dataset,
    split_train_test,
    ensure_feature_order_and_fill,
    apply_standardization,
    local_sums_for_scaler,
    ScalerStats,
)
from TabNet.server.config import SERVER_ADDRESS


PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Utilities: model params
# ----------------------------
def get_model_params(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_params(model: nn.Module, params: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(params):
        raise ValueError(f"Param mismatch: got {len(params)} arrays, expected {len(keys)}")
    new_state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(new_state, strict=True)


# ----------------------------
# Loss
# ----------------------------
def make_loss(cfg: Dict) -> nn.Module:
    name = cfg.get("LOSS_NAME", "mae").lower()
    if name == "huber":
        beta = float(cfg.get("HUBER_BETA", 2.0))
        return nn.SmoothL1Loss(beta=beta)
    return nn.L1Loss()


# ----------------------------
# Training / eval helpers
# ----------------------------
def train_one_client(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    loss_fn: nn.Module,
):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()


def eval_real_mae(
    model: nn.Module,
    X: np.ndarray,
    y_real: np.ndarray,
    clip_min: float,
    clip_max: float,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> float:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        pred_norm = model(xb).detach().cpu().numpy()

    pred = pred_norm * y_std + y_mean
    pred = np.clip(pred, clip_min, clip_max)
    return float(mean_absolute_error(y_real, pred))


# ----------------------------
# Batch size "dinamica intelligente"
# 1) cap su len(ds)
# 2) >=2 (BatchNorm training richiede almeno 2)
# 3) prova a evitare resto=1 riducendo bs finché possibile
# ----------------------------
def choose_smart_batch_size(n_samples: int, cfg_bs: int) -> int:
    if n_samples <= 0:
        return 2

    # caso estremo: un solo sample
    if n_samples == 1:
        return 2

    bs = int(cfg_bs)
    bs = max(bs, 2)
    bs = min(bs, n_samples)

    # se bs == n_samples -> un singolo batch, ok (n_samples>=2)
    if bs == n_samples:
        return bs

    # Evita resto=1: se n_samples % bs == 1, prova a ridurre bs
    # per trovare un valore che non produca resto 1, senza scendere sotto 2.
    if n_samples % bs == 1:
        candidate = bs
        while candidate > 2 and (n_samples % candidate == 1):
            candidate -= 1
        bs = max(2, candidate)

    return bs


# ----------------------------
# Batch sampler "zero data loss"
# Se l'ultimo batch ha size=1, lo unisce al batch precedente.
# (Quindi non c'è mai un forward con batch=1)
# ----------------------------
class MinTwoBatchSampler(Sampler[List[int]]):
    def __init__(self, data_source, batch_size: int, shuffle: bool = True):
        self.data_source = data_source
        self.batch_size = max(2, int(batch_size))
        self.shuffle = bool(shuffle)

    def __iter__(self):
        n = len(self.data_source)
        if n == 0:
            return iter([])

        # Caso estremo: n==1, duplichiamo l'unico indice
        if n == 1:
            return iter([[0, 0]])

        sampler = RandomSampler(self.data_source) if self.shuffle else range(n)

        batches: List[List[int]] = []
        batch: List[int] = []

        for idx in sampler:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        # Se ultimo batch size=1, unisci al penultimo (zero data loss)
        if len(batches) >= 2 and len(batches[-1]) == 1:
            batches[-2].extend(batches[-1])
            batches.pop()

        # Paracadute: se rimane comunque un batch di 1 (caso raro), duplicalo
        if len(batches) == 1 and len(batches[0]) == 1:
            batches[0].append(batches[0][0])

        return iter(batches)

    def __len__(self):
        n = len(self.data_source)
        if n <= 1:
            return 1
        return math.ceil(n / self.batch_size)


# ----------------------------
# Flower client
# ----------------------------
class NNClient(NumPyClient):
    def __init__(self, cid: int, data_path: str, cfg: Dict):
        self.cid = int(cid)
        self.data_path = data_path
        self.cfg = cfg

        self.logger = logging.getLogger(f"client_{self.cid}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(LOGS_DIR / f"client_{self.cid}.log")
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.logger.addHandler(handler)

        X_df, y_sr = load_csv_dataset(data_path, label_col="label")
        self.feature_names_local = list(X_df.columns)

        X_tr_df, X_te_df, y_tr, y_te = split_train_test(
            X_df,
            y_sr,
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

        self.logger.info(
            f"Loaded {data_path} | X_train={self.X_train_df.shape}, X_test={self.X_test_df.shape}"
        )

    def _build_model_if_needed(self, input_dim: int):
        if self.model is None:
            tabcfg = TabNetConfig(
                n_d=int(self.cfg["TABNET_N_D"]),
                n_a=int(self.cfg["TABNET_N_A"]),
                n_steps=int(self.cfg["TABNET_N_STEPS"]),
                gamma=float(self.cfg["TABNET_GAMMA"]),
                n_shared=int(self.cfg["TABNET_N_SHARED"]),
                n_independent=int(self.cfg["TABNET_N_INDEPENDENT"]),
                bn_virtual_bs=int(self.cfg["TABNET_BN_VIRTUAL_BS"]),
                bn_momentum=float(self.cfg["TABNET_BN_MOMENTUM"]),
            )
            self.model = TabNetRegressor(input_dim=input_dim, cfg=tabcfg).to(self.device)

    def get_parameters(self, config):
        if self.model is None:
            return []
        return get_model_params(self.model)

    def fit(self, parameters, config):
        phase = (config or {}).get("phase", "train")

        # ROUND 1: invio stats per scaler globale + y stats
        if phase == "scaler":
            feat_names = self.feature_names_local
            n, s, ssq = local_sums_for_scaler(self.X_train_df.fillna(0.0), feat_names)

            ytr64 = self.y_train.astype(np.float64)
            metrics = {
                "feature_names": json.dumps(feat_names),
                "n": int(n),
                "sum": json.dumps(s.tolist()),
                "sumsq": json.dumps(ssq.tolist()),
                "y_n": int(ytr64.shape[0]),
                "y_sum": float(np.sum(ytr64)),
                "y_sumsq": float(np.sum(ytr64 * ytr64)),
            }
            self.logger.info("Phase=scaler: sent X sums + Y sums")
            return [], n, metrics

        # ROUND >=2: training
        gf = json.loads(config["global_features"])
        mean = np.array(json.loads(config["scaler_mean"]), dtype=np.float32)
        std = np.array(json.loads(config["scaler_std"]), dtype=np.float32)

        y_mean = float(config.get("y_mean", 0.0))
        y_std = float(config.get("y_std", 1.0))
        if y_std <= 1e-12:
            y_std = 1.0

        self.global_features = list(gf)
        self.scaler = ScalerStats(mean=mean, std=std)

        Xtr = ensure_feature_order_and_fill(self.X_train_df.copy(), self.global_features)
        Xte = ensure_feature_order_and_fill(self.X_test_df.copy(), self.global_features)
        Xtr = apply_standardization(Xtr, self.scaler)
        Xte = apply_standardization(Xte, self.scaler)

        self._build_model_if_needed(input_dim=Xtr.shape[1])

        if parameters and len(parameters) > 0:
            set_model_params(self.model, parameters)

        ytr_norm = (self.y_train - y_mean) / y_std

        ds = TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(ytr_norm, dtype=torch.float32),
        )

        cfg_bs = int(self.cfg["BATCH_SIZE"])
        smart_bs = choose_smart_batch_size(len(ds), cfg_bs)

        batch_sampler = MinTwoBatchSampler(ds, batch_size=smart_bs, shuffle=True)
        loader = DataLoader(ds, batch_sampler=batch_sampler)

        self.logger.info(
            f"Train samples={len(ds)} | cfg_bs={cfg_bs} | smart_bs={smart_bs} | "
            f"epochs={int(self.cfg['LOCAL_EPOCHS'])}"
        )

        loss_fn = make_loss(self.cfg)

        train_one_client(
            self.model,
            loader,
            epochs=int(self.cfg["LOCAL_EPOCHS"]),
            lr=float(self.cfg["LR"]),
            weight_decay=float(self.cfg["WEIGHT_DECAY"]),
            device=self.device,
            loss_fn=loss_fn,
        )

        mae_train = eval_real_mae(
            self.model,
            Xtr,
            self.y_train,
            clip_min=float(self.cfg["CLIP_MIN"]),
            clip_max=float(self.cfg["CLIP_MAX"]),
            device=self.device,
            y_mean=y_mean,
            y_std=y_std,
        )

        return get_model_params(self.model), len(Xtr), {"train_mae_real": float(mae_train)}

    def evaluate(self, parameters, config):
        if self.global_features is None or self.scaler is None or self.model is None:
            return float("nan"), 0, {"eval_mae_real": float("nan")}

        if parameters and len(parameters) > 0:
            set_model_params(self.model, parameters)

        y_mean = float(config.get("y_mean", 0.0))
        y_std = float(config.get("y_std", 1.0))
        if y_std <= 1e-12:
            y_std = 1.0

        Xte = ensure_feature_order_and_fill(self.X_test_df.copy(), self.global_features)
        Xte = apply_standardization(Xte, self.scaler)

        mae = eval_real_mae(
            self.model,
            Xte,
            self.y_test,
            clip_min=float(self.cfg["CLIP_MIN"]),
            clip_max=float(self.cfg["CLIP_MAX"]),
            device=self.device,
            y_mean=y_mean,
            y_std=y_std,
        )
        return float(mae), len(Xte), {"eval_mae_real": float(mae)}


def main():
    if len(sys.argv) < 3:
        print("Usage: python client_app.py <client_id> <csv_path>")
        raise SystemExit(1)

    client_id = int(sys.argv[1])
    data_path = sys.argv[2]
    if not os.path.exists(data_path):
        print(f"ERRORE: Data file not found: {data_path}")
        raise SystemExit(1)

    import client_params as P

    cfg = {
        "CLIP_MIN": P.CLIP_MIN,
        "CLIP_MAX": P.CLIP_MAX,
        "TEST_SIZE": P.TEST_SIZE,
        "RANDOM_STATE": P.RANDOM_STATE,
        "SHUFFLE_SPLIT": P.SHUFFLE_SPLIT,
        "LOCAL_EPOCHS": P.LOCAL_EPOCHS,
        "BATCH_SIZE": P.BATCH_SIZE,
        "LR": P.LR,
        "WEIGHT_DECAY": P.WEIGHT_DECAY,
        "TABNET_N_D": P.TABNET_N_D,
        "TABNET_N_A": P.TABNET_N_A,
        "TABNET_N_STEPS": P.TABNET_N_STEPS,
        "TABNET_GAMMA": P.TABNET_GAMMA,
        "TABNET_N_SHARED": P.TABNET_N_SHARED,
        "TABNET_N_INDEPENDENT": P.TABNET_N_INDEPENDENT,
        "TABNET_BN_VIRTUAL_BS": P.TABNET_BN_VIRTUAL_BS,
        "TABNET_BN_MOMENTUM": P.TABNET_BN_MOMENTUM,
        "LOSS_NAME": P.LOSS_NAME,
        "HUBER_BETA": P.HUBER_BETA,
    }

    client = NNClient(client_id, data_path, cfg)
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)


if __name__ == "__main__":
    main()
