# preprocess_fed_mlp.py
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pathlib import Path

from dataset.dataset_cfg import HOLDOUT

# Project root = directory dove si trova questo file
PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------
# Config facile da modificare
# -----------------------------
@dataclass
class PreprocessConfig:
    # Path
    raw_root: Path = PROJECT_ROOT/"../../dataset/raw_dataset"
    out_root: Path = PROJECT_ROOT/"clients_dataset"
    x_test_path: Path = PROJECT_ROOT/"../../dataset/raw_dataset/x_test.csv"

    # Federated setup
    n_groups: int = 9
    holdout: Optional[int] = None  # None oppure 0..8

    # CSV parsing
    sep: str = ";"
    encoding: str = "utf-8"

    # Colonne
    label_col: str = "label"
    drop_cols_if_exist: Tuple[str, ...] = ("Unnamed: 0", "day")

    # Time-series
    time_series_cols: Tuple[str, ...] = ("hr_time_series", "resp_time_series", "stress_time_series")
    drop_original_time_series: bool = True

    # Cleaning / invalid handling
    coerce_numeric: bool = True  # prova a convertire object->numeric (dopo aver tolto time-series)
    replace_infs: bool = True

    # Row filtering (SOLO per train-mode: non x_test, non holdout)
    enable_row_drop: bool = True
    row_drop_max_nan_frac: float = 0.80  # droppa righe con >30% NaN sulle feature
    row_drop_if_all_nan: bool = True

    # Imputation
    enable_impute: bool = True
    impute_strategy: str = "median"  # "median" o "mean"

    # Outlier clipping (winsorization) - fit su train-scope
    enable_clip: bool = True
    clip_low_q: float = 0.01
    clip_high_q: float = 0.99

    # Scaling - fit su train-scope
    enable_scale: bool = False
    scaler_type: str = "standard"  # "standard" o "robust"
    fit_scope: str = "per_client"      # "global" o "per_client"
    # se "global": parametri stimati su tutti i client TRAIN (escluso holdout se impostato)
    # se "per_client": ogni client ha il suo scaler (x_test userà GLOBAL se disponibile, altrimenti nessuno)

    # Salvataggio
    float_format: str = "%.6g"

    sort_by_col: str | None = None
    sort_ascending: bool = True

    # Feature engineering (safe w/o scaling)
    enable_ts_derived: bool = True

    enable_log1p: bool = True
    log1p_cols: Tuple[str, ...] = (
        "sleep_deepSleepSeconds",
        "sleep_lightSleepSeconds",
        "sleep_remSleepSeconds",
        "sleep_awakeSleepSeconds",
        "sleep_unmeasurableSleepSeconds",
        "act_totalCalories",
        "act_activeKilocalories",
        "act_distance",
        "act_activeTime",
    )

    enable_poly: bool = True
    poly_degree: int = 2
    poly_cols: Tuple[str, ...] = (
        # bounded-ish columns: hr/stress/resp/sleep rates
        "hr_maxHeartRate",
        "hr_minHeartRate",
        "hr_restingHeartRate",
        "str_maxStressLevel",
        "str_avgStressLevel",
        "resp_lowestRespirationValue",
        "resp_highestRespirationValue",
        "sleep_averageRespirationValue",
        "sleep_avgHeartRate",
    )


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path, cfg: PreprocessConfig) -> pd.DataFrame:
    return pd.read_csv(path, sep=cfg.sep, encoding=cfg.encoding, engine="python")


def safe_literal_list(x) -> Optional[List[float]]:
    """
    Converte stringhe tipo "[1, 2, 3]" in lista.
    Ritorna None se non parsabile.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return None

    s = x.strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    # prova literal_eval
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            out = []
            for e in v:
                try:
                    out.append(float(e))
                except Exception:
                    # se un elemento è non numerico, ignora
                    continue
            return out if len(out) > 0 else None
        return None
    except Exception:
        return None


def ts_features(values: Optional[List[float]], prefix: str) -> Dict[str, float]:
    """
    Estrae feature robuste da una time-series.
    """
    feats: Dict[str, float] = {}
    if values is None or len(values) == 0:
        keys = ["len", "mean", "std", "min", "max", "median", "p10", "p90", "slope"]
        for k in keys:
            feats[f"{prefix}__{k}"] = np.nan
        return feats

    a = np.array(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return ts_features(None, prefix)

    feats[f"{prefix}__len"] = float(a.size)
    feats[f"{prefix}__mean"] = float(np.mean(a))
    feats[f"{prefix}__std"] = float(np.std(a))
    feats[f"{prefix}__min"] = float(np.min(a))
    feats[f"{prefix}__max"] = float(np.max(a))
    feats[f"{prefix}__median"] = float(np.median(a))
    feats[f"{prefix}__p10"] = float(np.quantile(a, 0.10))
    feats[f"{prefix}__p90"] = float(np.quantile(a, 0.90))

    # slope lineare semplice su indice (trend)
    if a.size >= 2:
        x = np.arange(a.size, dtype=float)
        x_mean = x.mean()
        y_mean = a.mean()
        denom = np.sum((x - x_mean) ** 2)
        slope = np.sum((x - x_mean) * (a - y_mean)) / denom if denom > 0 else 0.0
        feats[f"{prefix}__slope"] = float(slope)
    else:
        feats[f"{prefix}__slope"] = 0.0

    return feats


def build_client_from_group(group_dir: Path, cfg: PreprocessConfig) -> pd.DataFrame:
    csvs = sorted(group_dir.glob("*.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError(f"Nessun CSV in {group_dir}")

    parts = [read_csv_safe(p, cfg) for p in csvs]
    df = pd.concat(parts, axis=0, ignore_index=True)

    if cfg.sort_by_col and cfg.sort_by_col in df.columns:
        df = df.sort_values(
            cfg.sort_by_col,
            ascending=cfg.sort_ascending
        ).reset_index(drop=True)

    return df

def add_feature_engineering(X: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    X2 = X.copy()

    # --- derived features dalle time-series aggregate
    if cfg.enable_ts_derived:
        for base in cfg.time_series_cols:
            cmin = f"{base}__min"
            cmax = f"{base}__max"
            cmean = f"{base}__mean"
            cstd = f"{base}__std"
            cp10 = f"{base}__p10"
            cp90 = f"{base}__p90"

            if cmin in X2.columns and cmax in X2.columns:
                X2[f"{base}__range"] = X2[cmax] - X2[cmin]
            if cp10 in X2.columns and cp90 in X2.columns:
                X2[f"{base}__iqr_10_90"] = X2[cp90] - X2[cp10]
            if cmean in X2.columns and cstd in X2.columns:
                eps = 1e-6
                X2[f"{base}__cv"] = X2[cstd] / (X2[cmean].abs() + eps)

    # --- log1p per feature positive (compressiva, ottima senza scaling)
    if cfg.enable_log1p:
        for c in cfg.log1p_cols:
            if c not in X2.columns:
                continue
            s = X2[c].astype(float)
            mask = s >= 0
            X2.loc[mask, f"{c}__log1p"] = np.log1p(s.loc[mask])

    # --- polynomial (safe subset)
    if cfg.enable_poly and cfg.poly_degree >= 2:
        for c in cfg.poly_cols:
            if c not in X2.columns:
                continue
            s = X2[c].astype(float)
            X2[f"{c}__pow2"] = s * s
            if cfg.poly_degree >= 3:
                X2[f"{c}__pow3"] = s * s * s

    return X2



# -----------------------------
# Preprocessing core
# -----------------------------
@dataclass
class FittedParams:
    feature_cols: List[str]                    # ordine finale colonne feature
    impute_values: Dict[str, float]            # col -> value
    clip_bounds: Dict[str, Tuple[float, float]]# col -> (low, high)
    scaler: Dict[str, Tuple[float, float]]     # col -> (center, scale) center=mean/median, scale=std/iqr


def compute_impute_values(X: pd.DataFrame, cfg: PreprocessConfig) -> Dict[str, float]:
    vals = {}
    for c in X.columns:
        s = X[c]
        if cfg.impute_strategy == "mean":
            v = float(np.nanmean(s.values))
        else:
            v = float(np.nanmedian(s.values))
        if not np.isfinite(v):
            v = 0.0
        vals[c] = v
    return vals


def apply_impute(X: pd.DataFrame, impute_values: Dict[str, float]) -> pd.DataFrame:
    X2 = X.copy()
    for c, v in impute_values.items():
        if c in X2.columns:
            X2[c] = X2[c].fillna(v)
    return X2


def compute_clip_bounds(X: pd.DataFrame, cfg: PreprocessConfig) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for c in X.columns:
        s = X[c].astype(float)
        lo = float(np.nanquantile(s.values, cfg.clip_low_q))
        hi = float(np.nanquantile(s.values, cfg.clip_high_q))
        if not np.isfinite(lo):
            lo = -np.inf
        if not np.isfinite(hi):
            hi = np.inf
        if lo > hi:
            lo, hi = hi, lo
        bounds[c] = (lo, hi)
    return bounds


def apply_clip(X: pd.DataFrame, clip_bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    X2 = X.copy()
    for c, (lo, hi) in clip_bounds.items():
        if c in X2.columns:
            X2[c] = X2[c].clip(lower=lo, upper=hi)
    return X2


def compute_scaler_params(X: pd.DataFrame, cfg: PreprocessConfig) -> Dict[str, Tuple[float, float]]:
    params = {}
    for c in X.columns:
        v = X[c].astype(float).values
        if cfg.scaler_type == "robust":
            center = float(np.nanmedian(v))
            q1 = float(np.nanquantile(v, 0.25))
            q3 = float(np.nanquantile(v, 0.75))
            scale = float(q3 - q1)
        else:
            center = float(np.nanmean(v))
            scale = float(np.nanstd(v))

        if not np.isfinite(center):
            center = 0.0
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        params[c] = (center, scale)
    return params


def apply_scale(X: pd.DataFrame, scaler_params: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    X2 = X.copy()
    for c, (center, scale) in scaler_params.items():
        if c in X2.columns:
            X2[c] = (X2[c].astype(float) - center) / scale
    return X2


def extract_time_series_features(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    df2 = df.copy()
    for col in cfg.time_series_cols:
        if col not in df2.columns:
            continue
        feats_rows = []
        for x in df2[col].values:
            values = safe_literal_list(x)
            feats_rows.append(ts_features(values, prefix=col))
        feats_df = pd.DataFrame(feats_rows, index=df2.index)
        df2 = pd.concat([df2, feats_df], axis=1)

        if cfg.drop_original_time_series:
            df2 = df2.drop(columns=[col])
    return df2


def basic_cleaning(df: pd.DataFrame, cfg: PreprocessConfig, is_test_mode: bool) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df2 = df.copy()

    # drop colonne inutili (es. index)
    for c in cfg.drop_cols_if_exist:
        if c in df2.columns:
            df2 = df2.drop(columns=[c])

    # separa y se presente (forza float)
    y = None
    if cfg.label_col in df2.columns:
        y = pd.to_numeric(df2[cfg.label_col], errors="coerce").astype("float32")
        df2 = df2.drop(columns=[cfg.label_col])

    # feature da time-series
    df2 = extract_time_series_features(df2, cfg)

    # converti a numerico se richiesto (ignora errori)
    if cfg.coerce_numeric:
        for c in df2.columns:
            if df2[c].dtype == "object":
                df2[c] = pd.to_numeric(df2[c], errors="coerce")

    df2 = add_feature_engineering(df2, cfg)

    # inf -> nan
    if cfg.replace_infs:
        df2 = df2.replace([np.inf, -np.inf], np.nan)

    # drop righe (solo train-mode)
    if (not is_test_mode) and cfg.enable_row_drop:
        if cfg.row_drop_if_all_nan:
            df2 = df2.dropna(axis=0, how="all")

        if cfg.row_drop_max_nan_frac is not None:
            nan_frac = df2.isna().mean(axis=1)
            keep = nan_frac <= cfg.row_drop_max_nan_frac
            df2 = df2.loc[keep].copy()
            if y is not None:
                y = y.loc[df2.index]

    df2 = df2.astype("float32")
    if y is not None:
        y = y.astype("float32")

    return df2, y


def fit_global_params(train_clients: Dict[int, pd.DataFrame],
                      cfg: PreprocessConfig) -> FittedParams:
    # concat tutte le feature train (senza label)
    X_all = pd.concat(list(train_clients.values()), axis=0, ignore_index=True)

    feature_cols = list(X_all.columns)

    # impute
    impute_values = compute_impute_values(X_all[feature_cols], cfg) if cfg.enable_impute else {}

    # per stimare clip/scaler in modo stabile, prima imputiamo temporaneamente
    X_tmp = X_all[feature_cols].copy()
    if cfg.enable_impute:
        X_tmp = apply_impute(X_tmp, impute_values)

    clip_bounds = compute_clip_bounds(X_tmp, cfg) if cfg.enable_clip else {}
    if cfg.enable_clip:
        X_tmp = apply_clip(X_tmp, clip_bounds)

    scaler_params = compute_scaler_params(X_tmp, cfg) if cfg.enable_scale else {}

    return FittedParams(
        feature_cols=feature_cols,
        impute_values=impute_values,
        clip_bounds=clip_bounds,
        scaler=scaler_params
    )


def transform_with_params(X: pd.DataFrame, params: FittedParams, cfg: PreprocessConfig) -> pd.DataFrame:
    # allinea colonne (aggiunge mancanti, rimuove extra)
    X2 = X.copy()
    for c in params.feature_cols:
        if c not in X2.columns:
            X2[c] = np.nan
    X2 = X2[params.feature_cols].copy()

    if cfg.enable_impute and params.impute_values:
        X2 = apply_impute(X2, params.impute_values)

    if cfg.enable_clip and params.clip_bounds:
        X2 = apply_clip(X2, params.clip_bounds)

    if cfg.enable_scale and params.scaler:
        X2 = apply_scale(X2, params.scaler)

    return X2


# -----------------------------
# Pipeline end-to-end
# -----------------------------
def main(cfg: PreprocessConfig) -> None:
    ensure_dir(cfg.out_root)

    # 1) Build merged clients
    merged_raw: Dict[int, pd.DataFrame] = {}
    for g in range(cfg.n_groups):
        group_dir = cfg.raw_root / f"group{g}"
        merged_raw[g] = build_client_from_group(group_dir, cfg)

    # 2) Clean split: train-mode vs test-mode (x_test e holdout)
    clients_X: Dict[int, pd.DataFrame] = {}
    clients_y: Dict[int, Optional[pd.Series]] = {}

    for g, df in merged_raw.items():
        is_holdout = (cfg.holdout is not None and g == cfg.holdout)
        Xg, yg = basic_cleaning(df, cfg, is_test_mode=is_holdout)  # holdout: test-mode
        clients_X[g] = Xg
        clients_y[g] = yg

    # x_test
    x_test_df = read_csv_safe(cfg.x_test_path, cfg)
    x_test_X, _ = basic_cleaning(x_test_df, cfg, is_test_mode=True)

    # 3) Fit params
    if cfg.fit_scope == "global":
        train_groups = [g for g in range(cfg.n_groups) if cfg.holdout is None or g != cfg.holdout]
        train_clients = {g: clients_X[g] for g in train_groups}
        global_params = fit_global_params(train_clients, cfg)
    else:
        global_params = None  # per-client scalers

    # 4) Transform + Save
    for g in range(cfg.n_groups):
        is_holdout = (cfg.holdout is not None and g == cfg.holdout)

        Xg = clients_X[g]
        yg = clients_y[g]

        # scegli parametri
        if cfg.fit_scope == "global":
            params = global_params
        else:
            # per-client: fit SOLO se non è holdout
            if is_holdout:
                params = None
            else:
                params = fit_global_params({g: Xg}, cfg)

        if params is not None:
            Xg_t = transform_with_params(Xg, params, cfg)
            Xg_t = Xg_t.astype("float32")
        else:
            # fallback: niente scaling/clipping/impute parametrici globali
            # ma per coerenza: se enable_impute, imputiamo "localmente" senza drop (già fatto in basic_cleaning)
            Xg_t = Xg.copy()
            if cfg.enable_impute:
                imp = compute_impute_values(Xg_t, cfg)
                Xg_t = apply_impute(Xg_t, imp)

        # ricompone label se presente
        if yg is not None:
            out_df = pd.concat([yg.reset_index(drop=True), Xg_t.reset_index(drop=True)], axis=1)
            out_df = out_df.rename(columns={0: cfg.label_col})
            if cfg.label_col != yg.name and yg.name is not None:
                # mantieni il nome originale
                out_df = out_df.rename(columns={yg.name: cfg.label_col})
        else:
            out_df = Xg_t.reset_index(drop=True)

        out_path = cfg.out_root / f"group{g}_merged_clean.csv"
        out_df.to_csv(out_path, index=False, sep=",", float_format=cfg.float_format)

    # x_test transform (FEDERATO): fit solo su x_test stesso
    x_test_params = fit_global_params({0: x_test_X}, cfg)  # usa la stessa funzione ma su un solo "client"
    x_test_t = transform_with_params(x_test_X, x_test_params, cfg)

    x_out_path = cfg.out_root / "x_test_clean.csv"
    x_test_t.to_csv(x_out_path, index=False, sep=",", float_format=cfg.float_format)

    print("DONE. Output in:", cfg.out_root)


if __name__ == "__main__":
    cfg = PreprocessConfig(
        holdout=HOLDOUT,
        scaler_type="standard",
        enable_clip=True,
        clip_low_q=0.01,
        clip_high_q=0.99,
        enable_row_drop=True,
    )
    main(cfg)

