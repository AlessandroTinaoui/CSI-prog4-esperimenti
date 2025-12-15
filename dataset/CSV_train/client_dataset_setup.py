from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


# -----------------------------
# Config
# -----------------------------
@dataclass
class CleanConfig:
    # nomi colonne nel tuo CSV
    label_col: str = "label"
    day_col: Optional[str] = "day"       # metti None se non vuoi considerarla

    # regole pulizia
    drop_label_zero: bool = True
    min_non_null_frac: float = 0.70      # "giornate con pochi dati"
    iqr_k: float = 1.5                   # outlier IQR
    winsorize_iqr: bool = True           # True=clip, False=NaN (poi imputo)

    # imputazione
    knn_k: int = 3
    drop_all_nan_features: bool = True   # elimina feature sempre NaN


import re

def drop_time_series_columns(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Rimuove colonne time-series:
    - per nome (keyword)
    - per euristica: colonne object con stringhe molto lunghe (time series serializzate)
    """
    out = df.copy()

    name_keywords = ["time_series", "timeseries", "_ts", "series"]
    cols_by_name = [c for c in out.columns if any(k in c.lower() for k in name_keywords)]

    # euristica su colonne object: stringhe mediamente molto lunghe
    object_cols = [c for c in out.columns if out[c].dtype == "object"]
    cols_by_len = []
    for c in object_cols:
        s = out[c].dropna()
        if s.empty:
            continue
        # calcola lunghezza media solo su stringhe
        s = s.astype(str)
        if s.map(len).mean() > 200:   # soglia: aumenta/diminuisci se serve
            cols_by_len.append(c)

    to_drop = sorted(set(cols_by_name + cols_by_len))
    if to_drop:
        out = out.drop(columns=to_drop, errors="ignore")

    return out


# -----------------------------
# Lettura CSV (pulita, senza magie)
# -----------------------------
def read_user_csv(path: str) -> pd.DataFrame:
    # I tuoi file sono tabelle separate da ';'
    df = pd.read_csv(path, sep=";")
    # spesso c'è una colonna indice inutile
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df


# -----------------------------
# Pulizia dataset singolo user
# -----------------------------
def _select_numeric_feature_cols(df: pd.DataFrame, cfg: CleanConfig) -> List[str]:
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)

    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _coerce_numeric_features(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)

    for c in out.columns:
        if c in exclude:
            continue
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def drop_invalid_labels(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    if cfg.label_col not in df.columns:
        raise ValueError(f"Colonna label '{cfg.label_col}' NON trovata. Colonne: {list(df.columns)}")

    out = df.dropna(subset=[cfg.label_col]).copy()
    if cfg.drop_label_zero:
        out = out[out[cfg.label_col] != 0]
    return out


def drop_low_info_days(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    feat = _select_numeric_feature_cols(out, cfg)
    if not feat:
        return out

    frac = out[feat].notna().mean(axis=1)
    out = out[frac >= cfg.min_non_null_frac].copy()
    return out


def handle_outliers_iqr(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    feat = _select_numeric_feature_cols(out, cfg)

    for c in feat:
        s = out[c]
        # se pochi valori, evita di fare IQR
        if s.dropna().shape[0] < 5:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lo = q1 - cfg.iqr_k * iqr
        hi = q3 + cfg.iqr_k * iqr

        if cfg.winsorize_iqr:
            out[c] = s.clip(lo, hi)
        else:
            out.loc[(s < lo) | (s > hi), c] = np.nan

    return out


def knn_impute(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    feat = _select_numeric_feature_cols(out, cfg)
    if not feat:
        return out

    if cfg.drop_all_nan_features:
        all_nan = [c for c in feat if out[c].isna().all()]
        if all_nan:
            out = out.drop(columns=all_nan)
            feat = [c for c in feat if c not in all_nan]
            if not feat:
                return out

    imputer = KNNImputer(n_neighbors=cfg.knn_k, weights="distance")
    out[feat] = imputer.fit_transform(out[feat].astype(float))
    return out


def clean_user_df(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()

    out = drop_time_series_columns(out, cfg)

    # opzionale: day come int (se è già numerico ok; se è stringa tipo "1", converte)
    if cfg.day_col and cfg.day_col in out.columns:
        out[cfg.day_col] = pd.to_numeric(out[cfg.day_col], errors="ignore")

    out = _coerce_numeric_features(out, cfg)
    out = drop_invalid_labels(out, cfg)
    out = drop_low_info_days(out, cfg)
    out = handle_outliers_iqr(out, cfg)
    out = knn_impute(out, cfg)

    # ordina per day se presente
    if cfg.day_col and cfg.day_col in out.columns:
        out = out.sort_values(cfg.day_col)

    return out.reset_index(drop=True)


# -----------------------------
# Merge per client (groupX)
# -----------------------------
def parse_user_id(filename: str) -> str:
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("_")
    if "user" in parts:
        i = parts.index("user")
        if i + 1 < len(parts):
            return parts[i + 1]
    return base


def build_clients(base_dir: str, out_dir: str, cfg: CleanConfig) -> List[Tuple[str, str]]:
    """
    base_dir contiene group0, group1, ...
    dentro ogni group: dataset_user_*_train.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    group_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, "group*")) if os.path.isdir(d)])
    if not group_dirs:
        raise FileNotFoundError(f"Nessuna cartella group* trovata in: {base_dir}")

    saved: List[Tuple[str, str]] = []

    for gdir in group_dirs:
        client_id = os.path.basename(gdir)  # group0...
        user_files = sorted(glob.glob(os.path.join(gdir, "*.csv")))
        if not user_files:
            print(f"[WARN] {client_id}: nessun csv trovato.")
            continue

        parts = []
        for p in user_files:
            user_id = parse_user_id(p)

            df = read_user_csv(p)
            df["client_id"] = client_id
            df["user_id"] = user_id
            df["source_file"] = os.path.basename(p)

            df = clean_user_df(df, cfg)
            parts.append(df)

        merged = pd.concat(parts, ignore_index=True)

        # opzionale: ordine stabile per (day, user)
        if cfg.day_col and cfg.day_col in merged.columns:
            merged = merged.sort_values([cfg.day_col, "user_id"]).reset_index(drop=True)

        out_path = os.path.join(out_dir, f"{client_id}_merged_clean.csv")
        merged.to_csv(out_path, index=False)

        print(f"[OK] {client_id}: {len(user_files)} utenti -> {merged.shape[0]} righe, {merged.shape[1]} colonne | {out_path}")
        saved.append((client_id, out_path))

    return saved


if __name__ == "__main__":
    # path robusti: partono dalla cartella dove sta questo script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Nel tuo caso: .../dataset/CSV_train/client_dataset_setup.py
    # e i group sono in: .../dataset/CSV_train/CSV_train/group0...
    BASE_DIR = os.path.join(SCRIPT_DIR, "CSV_train")
    OUT_DIR = os.path.join(SCRIPT_DIR, "CSV_train_clean")

    cfg = CleanConfig(
        label_col="label",
        day_col="day",
        min_non_null_frac=0.70,
        knn_k=3,
        iqr_k=1.5,
        winsorize_iqr=False
    )

    build_clients(BASE_DIR, OUT_DIR, cfg)
