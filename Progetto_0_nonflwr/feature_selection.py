from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

from .config import DataConfig, FeatureSelectionConfig, RFLocalConfig


def get_feature_columns(df: pd.DataFrame, dc: DataConfig) -> List[str]:
    drop = set(dc.drop_cols)
    drop.add(dc.label_col)
    if dc.day_col and dc.day_col in df.columns:
        drop.add(dc.day_col)
    return [c for c in df.columns if c not in drop]


def make_xy(df: pd.DataFrame, dc: DataConfig, features: List[str]):
    X = df[features]
    y = df[dc.label_col].astype(float)
    return X, y


def local_permutation_importance(
    df: pd.DataFrame,
    dc: DataConfig,
    rf_cfg: RFLocalConfig,
    fs_cfg: FeatureSelectionConfig,
    features: List[str],
) -> Dict[str, float]:
    X, y = make_xy(df, dc, features)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=fs_cfg.val_size,
        random_state=fs_cfg.random_state
    )

    model = RandomForestRegressor(
        n_estimators=rf_cfg.n_estimators,
        max_depth=rf_cfg.max_depth,
        min_samples_leaf=rf_cfg.min_samples_leaf,
        n_jobs=rf_cfg.n_jobs,
        random_state=rf_cfg.random_state,
    )
    model.fit(X_tr, y_tr)

    pi = permutation_importance(
        model, X_val, y_val,
        scoring=fs_cfg.scoring,
        n_repeats=fs_cfg.n_repeats,
        random_state=fs_cfg.random_state,
        n_jobs=rf_cfg.n_jobs
    )

    return {f: float(v) for f, v in zip(features, pi.importances_mean)}


def aggregate_importances_weighted(
    payloads: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
    feats = list(payloads[0][1].keys())
    num = {f: 0.0 for f in feats}
    den = 0.0

    for n, imp in payloads:
        den += n
        for f in feats:
            num[f] += n * float(imp[f])

    return {f: (num[f] / den if den else 0.0) for f in feats}


def select_top_k(global_importances: Dict[str, float], k: int) -> List[str]:
    return [f for f, _ in sorted(global_importances.items(), key=lambda x: x[1], reverse=True)[:k]]
