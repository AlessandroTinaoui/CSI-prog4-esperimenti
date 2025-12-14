from __future__ import annotations
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_client_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def feature_cols(df: pd.DataFrame, label_col="label", day_col="day"):
    drop = {label_col, "client_id", "user_id", "source_file"}
    if day_col and day_col in df.columns:
        drop.add(day_col)
    return [c for c in df.columns if c not in drop]


def make_xy(df: pd.DataFrame, feats, label_col="label"):
    X = df[feats]
    y = df[label_col].astype(float)
    return X, y


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def local_perm_importance(
    df: pd.DataFrame,
    feats: list[str],
    label_col="label",
    scoring="neg_root_mean_squared_error",
    n_repeats=5,
    val_size=0.2,
    seed=42,
    rf_n_estimators=80,
    rf_max_depth=None,
    rf_min_samples_leaf=1,
):
    X, y = make_xy(df, feats, label_col)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=val_size, random_state=seed)

    model = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(Xtr, ytr)

    pi = permutation_importance(
        model, Xva, yva,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    return {f: float(v) for f, v in zip(feats, pi.importances_mean)}


def train_local_rf_estimators(
    df: pd.DataFrame,
    feats: list[str],
    label_col="label",
    seed=42,
    n_estimators=80,
    max_depth=None,
    min_samples_leaf=1,
):
    X, y = make_xy(df, feats, label_col)
    m = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=seed,
    )
    m.fit(X, y)
    return list(m.estimators_)  # lista di alberi


def build_global_model_from_estimators(estimators: list, n_features: int, seed=42):
    # fit dummy per inizializzare attributi sklearn
    m = RandomForestRegressor(n_estimators=1, n_jobs=-1, random_state=seed)
    Xd = np.zeros((2, n_features), dtype=float)
    yd = np.zeros((2,), dtype=float)
    m.fit(Xd, yd)

    m.estimators_ = estimators
    m.n_estimators = len(estimators)
    m.n_features_in_ = n_features
    m.n_outputs_ = 1
    return m


# --- serializzazione in uint8 per Flower ---
def dumps_to_uint8(obj) -> np.ndarray:
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return np.frombuffer(b, dtype=np.uint8)

def loads_from_uint8(arr: np.ndarray):
    b = arr.tobytes()
    return pickle.loads(b)

def dumps_json_uint8(obj) -> np.ndarray:
    b = json.dumps(obj).encode("utf-8")
    return np.frombuffer(b, dtype=np.uint8)

def loads_json_uint8(arr: np.ndarray):
    return json.loads(arr.tobytes().decode("utf-8"))
