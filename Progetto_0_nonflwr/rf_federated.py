from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import DataConfig, RFLocalConfig



def make_xy(df: pd.DataFrame, dc: DataConfig, features: List[str]):
    X = df[features]
    y = df[dc.label_col].astype(float)
    return X, y


def train_local_rf(df: pd.DataFrame, dc: DataConfig, rf_cfg: RFLocalConfig, features: List[str]) -> RandomForestRegressor:
    X, y = make_xy(df, dc, features)
    m = RandomForestRegressor(
        n_estimators=rf_cfg.n_estimators,
        max_depth=rf_cfg.max_depth,
        min_samples_leaf=rf_cfg.min_samples_leaf,
        n_jobs=rf_cfg.n_jobs,
        random_state=rf_cfg.random_state,
    )
    m.fit(X, y)
    return m




def build_global_forest(
    rf_cfg: RFLocalConfig,
    estimators: list,
    n_features: int,
    feature_names: Optional[List[str]] = None,
) -> RandomForestRegressor:
    """
    Modello globale = insieme di alberi locali (bagging federato).
    sklearn vuole che il modello sia "fitted" (attributi come n_outputs_).
    Quindi: fit dummy -> poi rimpiazziamo estimators_.
    """
    if len(estimators) == 0:
        raise ValueError("Nessun estimatore ricevuto: global forest vuota.")

    # 1) inizializzo e faccio un fit dummy per settare tutti gli attributi interni
    m = RandomForestRegressor(
        n_estimators=1,  # dummy
        max_depth=rf_cfg.max_depth,
        min_samples_leaf=rf_cfg.min_samples_leaf,
        n_jobs=rf_cfg.n_jobs,
        random_state=rf_cfg.random_state,
    )
    X_dummy = np.zeros((2, n_features), dtype=float)
    y_dummy = np.zeros((2,), dtype=float)
    m.fit(X_dummy, y_dummy)

    # 2) sostituisco gli alberi con quelli aggregati
    m.estimators_ = estimators
    m.n_estimators = len(estimators)

    # 3) set coerente dei campi principali
    m.n_features_in_ = n_features
    m.n_outputs_ = 1

    if feature_names is not None:
        m.feature_names_in_ = np.array(feature_names, dtype=object)

    return m


def eval_regression(model, df: pd.DataFrame, dc: DataConfig, features: List[str]) -> dict:
    X, y = make_xy(df, dc, features)
    yhat = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mae = float(mean_absolute_error(y, yhat))
    return {"rmse": rmse, "mae": mae}
