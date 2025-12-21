from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from extract_ts_features import extract_ts_features, TSFeatureConfig


# -----------------------------
# Config
# -----------------------------
@dataclass
class CleanConfig:
    label_col: str = "label"
    day_col: Optional[str] = "day"

    drop_label_zero: bool = True
    min_non_null_frac: float = 0.40

    iqr_k: float = 1.5

    # ✅ strategia esplicita per outlier tabellari
    outlier_strategy: str = "clean_col"
    # "clean_col" = NON tocca la colonna originale

    use_ts_features: bool = True
    debug: bool = True           # ✅ stampa info
    mode: str = "train"  # "train" oppure "infer"

    # --- NaN imputation (metodi slide) ---
    nan_method: str = "linear"          # "linear" (come richiesto)
    nan_fill_limit: Optional[int] = 3   # riempi max 3 NaN consecutivi (consigliato)



# -----------------------------
# Lettura CSV
# -----------------------------
def read_user_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


# -----------------------------
# Utility
# -----------------------------
def _select_numeric_feature_cols(df: pd.DataFrame, cfg: CleanConfig) -> List[str]:
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

def _select_numeric_original_cols_for_low_info(df: pd.DataFrame, cfg: CleanConfig) -> List[str]:
    cols = _select_numeric_feature_cols(df, cfg)
    return [
        c for c in cols
        if not (c.startswith("ts__") or c.endswith("__clean") or c.endswith("__is_outlier"))
    ]




def _coerce_numeric_features(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)

    for c in out.columns:
        if c not in exclude:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def impute_missing_values(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()

    # imputiamo SOLO le colonne numeriche "originali" (non ts__, non __clean, non flag)
    feat = _select_numeric_original_cols_for_low_info(out, cfg)
    if not feat:
        return out

    # importantissimo: per interpolazione serve ordine temporale
    if cfg.day_col and cfg.day_col in out.columns:
        out = out.sort_values(cfg.day_col)

    if cfg.nan_method == "linear":
        out[feat] = out[feat].interpolate(
            method="linear",
            limit=cfg.nan_fill_limit,
            limit_direction="both",
        )
    elif cfg.nan_method == "ffill":
        out[feat] = out[feat].ffill(limit=cfg.nan_fill_limit)
    elif cfg.nan_method == "bfill":
        out[feat] = out[feat].bfill(limit=cfg.nan_fill_limit)
    else:
        raise ValueError(f"Unknown nan_method: {cfg.nan_method}")

    return out

def fill_remaining_nans(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()

    # riempiamo SOLO colonne numeriche (escludendo id/label/day)
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)

    num_cols = [
        c for c in out.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(out[c])
        and not c.endswith("__is_outlier")  # flag booleani: non vanno imputati
    ]

    if not num_cols:
        return out

    # mediana per colonna (se una colonna è tutta NaN → mediana NaN)
    med = out[num_cols].median(numeric_only=True)

    # fallback: se mediana NaN, metti 0
    med = med.fillna(0.0)

    out[num_cols] = out[num_cols].fillna(med)
    return out

# -----------------------------
# Pulizia
# -----------------------------
def drop_invalid_labels(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    # In inferenza/test la label può non esserci
    if cfg.label_col not in df.columns:
        return df

    out = df.dropna(subset=[cfg.label_col]).copy()
    if cfg.drop_label_zero:
        out = out[out[cfg.label_col] != 0]
    return out



def drop_low_info_days(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    feat = _select_numeric_original_cols_for_low_info(df, cfg)
    if not feat:
        return df
    frac = df[feat].notna().mean(axis=1)
    return df[frac >= cfg.min_non_null_frac].copy()


def handle_outliers_iqr(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    if cfg.outlier_strategy != "clean_col":
        raise ValueError("Currently only outlier_strategy='clean_col' is supported.")

    out = df.copy()
    feat = _select_numeric_original_cols_for_low_info(out, cfg)

    for c in feat:
        s = out[c]

        # inizializza colonne anche se non abbastanza dati
        out[f"{c}__is_outlier"] = False
        out[f"{c}__clean"] = s

        if s.dropna().shape[0] < 5:
            continue

        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue

        lo, hi = q1 - cfg.iqr_k * iqr, q3 + cfg.iqr_k * iqr
        mask = (s < lo) | (s > hi)

        # ✅ NON tocchi la colonna originale
        out[f"{c}__is_outlier"] = mask.fillna(False)
        median_non_outlier = s[~mask].median()
        out[f"{c}__clean"] = s.where(~mask, median_non_outlier)

    return out



# -----------------------------
# Pipeline singolo user
# -----------------------------
def clean_user_df(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    n_rows_start = len(out)

    if cfg.mode not in {"train", "infer"}:
        raise ValueError(f"cfg.mode must be 'train' or 'infer', got: {cfg.mode}")

    # 1️⃣ Time series → feature
    if cfg.use_ts_features:
        ts_cfg = TSFeatureConfig(
            ts_cols=None,  # lascia None per inferenza automatica
            drop_original_ts_cols=True,
            drop_negative_values=True,  # ✅ sempre (train e test)

            # ✅ nuove regole: se troppi negativi => TS ignorata (used=0, feature NaN)
            add_quality_features=True,
            max_neg_frac_raw=0.50,  # es: >50% negativi nella raw => non usare la TS
            min_valid_points=5  # min punti validi dopo pulizia (negativi rimossi)
        )

        # DEBUG: quali colonne TS trova davvero
        if cfg.debug:
            from extract_ts_features import infer_ts_columns
            guessed = infer_ts_columns(out, ts_cfg)
            print("    TS cols inferred:", guessed)

        before_cols = set(out.columns)
        out = extract_ts_features(out, ts_cfg)
        after_cols = set(out.columns)

        ts_features = [c for c in after_cols - before_cols if c.startswith("ts__")]

        if cfg.debug:
            print(f"    TS features create: {len(ts_features)}")

    # 2️⃣ conversione numerica
    if cfg.day_col and cfg.day_col in out.columns:
        out[cfg.day_col] = pd.to_numeric(out[cfg.day_col], errors="coerce")

    out = _coerce_numeric_features(out, cfg)


    out = impute_missing_values(out, cfg)

    # 3️⃣ pulizia righe (solo training)
    if cfg.mode == "train":
        out = drop_invalid_labels(out, cfg)
        out = drop_low_info_days(out, cfg)

    # 4️⃣ outlier → NaN
    out = handle_outliers_iqr(out, cfg)

    out = fill_remaining_nans(out, cfg)

    if cfg.debug:
        n_nan = out.isna().sum().sum()
        print(
            f"    Rows: {n_rows_start} → {len(out)} | "
            f"Columns: {out.shape[1]} | "
            f"Total NaN: {int(n_nan)}"
        )

    return out.reset_index(drop=True)


# -----------------------------
# Merge per client
# -----------------------------
def parse_user_id(filename: str) -> str:
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("_")
    return parts[parts.index("user") + 1] if "user" in parts else base


def build_clients(base_dir: str, out_dir: str, cfg: CleanConfig):
    os.makedirs(out_dir, exist_ok=True)

    group_dirs = sorted(d for d in glob.glob(os.path.join(base_dir, "group*")) if os.path.isdir(d))
    print(base_dir)
    print(f"\n🔍 Client trovati: {len(group_dirs)}\n")

    for gdir in group_dirs:
        client_id = os.path.basename(gdir)
        user_files = sorted(glob.glob(os.path.join(gdir, "*.csv")))
        print(f"▶ Client {client_id} | utenti: {len(user_files)}")

        dfs = []
        for p in user_files:
            df = read_user_csv(p)
            df["client_id"] = client_id
            df["user_id"] = parse_user_id(p)
            df["source_file"] = os.path.basename(p)

            df = clean_user_df(df, cfg)
            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)
        if cfg.day_col and cfg.day_col in merged.columns:
            merged = merged.sort_values([cfg.day_col, "user_id"])

        out_path = os.path.join(out_dir, f"{client_id}_merged_clean.csv")
        merged.to_csv(out_path, index=False)

        print(
            f"✔ SALVATO {client_id}: "
            f"{merged.shape[0]} righe | {merged.shape[1]} colonne\n"
        )

def build_x_test(x_test_path: str, out_path: str, cfg: CleanConfig):
    df = read_user_csv(x_test_path)

    # se nel test non hai client/user id, li mettiamo fissi (non serve altro)
    df["client_id"] = "test"
    df["user_id"] = "test"
    df["source_file"] = os.path.basename(x_test_path)

    df_clean = clean_user_df(df, cfg)
    df_clean.to_csv(out_path, index=False)

    print(f"✔ SALVATO X_TEST: {df_clean.shape[0]} righe | {df_clean.shape[1]} colonne -> {out_path}")



if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # = .../dataset

    # =================
    # TRAIN (a gruppi)
    # =================
    TRAIN_BASE_DIR = os.path.join(SCRIPT_DIR, "../raw_dataset")  # <- vedi la tua struttura annidata
    TRAIN_OUT_DIR  = os.path.join(SCRIPT_DIR, "clients_dataset")

    cfg_train = CleanConfig(
        label_col="label",
        day_col="day",
        min_non_null_frac=0.40,
        debug=True,
        mode="train",
        outlier_strategy="clean_col",
        nan_method="linear",
        nan_fill_limit=3
    )
    build_clients(TRAIN_BASE_DIR, TRAIN_OUT_DIR, cfg_train)

    # =================
    # X_TEST (file singolo)
    # =================
    X_TEST_PATH = os.path.join(SCRIPT_DIR, "../raw_dataset/x_test.csv")
    X_TEST_OUT  = os.path.join(SCRIPT_DIR, "x_test_clean.csv")

    cfg_test = CleanConfig(
        label_col="label",   # anche se non c’è nel test, non fa nulla
        day_col="day",
        debug=True,
        mode="infer",        # ✅ mai drop righe
        outlier_strategy="clean_col",
        nan_method="linear",
        nan_fill_limit=3
    )
    build_x_test(X_TEST_PATH, X_TEST_OUT, cfg_test)
