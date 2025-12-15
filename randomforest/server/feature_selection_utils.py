# feature_selection_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


@dataclass
class L1Stats:
    n: int
    missing_rate: Dict[str, float]
    variance: Dict[str, float]
    feature_names: List[str]


def _get_X_y(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
    selected_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    drop_cols = drop_cols or []
    X = df.drop(columns=drop_cols + [target_col], errors="ignore")

    # Mantieni solo numeriche (coerente con il tuo codice attuale)
    X = X.select_dtypes(include=[np.number]).copy()

    if selected_features:
        keep = [c for c in selected_features if c in X.columns]
        X = X[keep].copy()

    y = df[target_col].values
    return X, y


def compute_l1_stats(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
) -> L1Stats:
    """Statistiche per Level 1: missing-rate e varianza (numeriche)."""
    X, _ = _get_X_y(df, target_col=target_col, drop_cols=drop_cols)

    missing_rate = X.isna().mean().to_dict()
    variance = pd.Series(np.nanvar(X.values, axis=0), index=X.columns).to_dict()

    return L1Stats(
        n=len(X),
        missing_rate=missing_rate,
        variance=variance,
        feature_names=list(X.columns),
    )


def compute_l2_permutation_importance(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
    task_type: str = "regression",
    n_estimators: int = 200,
    n_repeats: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Permutation importance su split random (NON time series).
    Ritorna Δloss per feature (positivo = importante).
    """
    X, y = _get_X_y(df, target_col=target_col, drop_cols=drop_cols)

    # Split random
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Imputazione semplice: mediana del train
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_val = X_val.fillna(med)

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        scoring = "accuracy"
        # permutation_importance restituisce drop in score (accuracy) => importanza positiva = peggiora l'accuracy
        # qui vogliamo sempre "più alto = più importante": va bene così
        sign = 1.0
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        scoring = "neg_mean_absolute_error"
        # con neg_MAE: quando peggiora, score diminuisce => importanza_mean positiva, ma in unità "neg_MAE"
        # convertiamo in ΔMAE positivo
        sign = -1.0

    model.fit(X_train, y_train)

    r = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )

    imp = sign * r.importances_mean  # ΔMAE (reg) o Δaccuracy (cls)
    return {feat: float(val) for feat, val in zip(X.columns, imp)}
