from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DROP_COLS_BASE = ["day", "client_id", "user_id", "source_file"]


@dataclass
class ScalerStats:
    mean: np.ndarray
    std: np.ndarray


def load_csv_dataset(
    csv_path: str,
    label_col: str = "label",
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path, sep=",")
    drop_cols = drop_cols or DROP_COLS_BASE
    drop_cols = [c for c in drop_cols if c in df.columns]

    if label_col not in df.columns:
        raise ValueError(f"Target column '{label_col}' non trovata in {csv_path}")

    X = df.drop(columns=drop_cols + [label_col], errors="ignore")
    y = df[label_col].astype(float)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)


def ensure_feature_order_and_fill(X: pd.DataFrame, global_features: List[str]) -> np.ndarray:
    for c in global_features:
        if c not in X.columns:
            X[c] = 0.0
    X = X[global_features].fillna(0.0)
    return X.to_numpy(dtype=np.float32)


def apply_standardization(X: np.ndarray, scaler: ScalerStats) -> np.ndarray:
    std = np.where(scaler.std <= 1e-12, 1.0, scaler.std)
    return (X - scaler.mean) / std


def local_sums_for_scaler(X: pd.DataFrame, feature_names: List[str]) -> Tuple[int, np.ndarray, np.ndarray]:
    arr = X[feature_names].to_numpy(dtype=np.float64)
    n = arr.shape[0]
    s = np.sum(arr, axis=0)
    ssq = np.sum(arr * arr, axis=0)
    return n, s, ssq
