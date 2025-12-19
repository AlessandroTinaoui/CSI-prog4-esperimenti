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
    winsorize_iqr: bool = False  # outlier -> NaN

    use_ts_features: bool = True
    debug: bool = True           # ✅ stampa info
    mode: str = "train"  # "train" oppure "infer"


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


def _coerce_numeric_features(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)

    for c in out.columns:
        if c not in exclude:
            out[c] = pd.to_numeric(out[c], errors="coerce")
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
    feat = _select_numeric_feature_cols(df, cfg)
    if not feat:
        return df
    frac = df[feat].notna().mean(axis=1)
    return df[frac >= cfg.min_non_null_frac].copy()


def handle_outliers_iqr(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    feat = _select_numeric_feature_cols(out, cfg)

    for c in feat:
        s = out[c]
        if s.dropna().shape[0] < 5:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        lo, hi = q1 - cfg.iqr_k * iqr, q3 + cfg.iqr_k * iqr
        out.loc[(s < lo) | (s > hi), c] = np.nan

    return out


# -----------------------------
# Pipeline singolo user
# -----------------------------
def clean_user_df(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    n_rows_start = len(out)

    # 1️⃣ Time series → feature
    if cfg.use_ts_features:
        ts_cfg = TSFeatureConfig(
            ts_cols=None,  # lascia None per inferenza automatica
            drop_original_ts_cols=True,
            drop_negative_values=True
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

    # 3️⃣ pulizia righe (solo training)
    if cfg.mode == "train":
        out = drop_invalid_labels(out, cfg)
        out = drop_low_info_days(out, cfg)

    # 4️⃣ outlier → NaN
    out = handle_outliers_iqr(out, cfg)

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


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    BASE_DIR = os.path.join(SCRIPT_DIR, "CSV_train")
    OUT_DIR = os.path.join(SCRIPT_DIR, "CSV_train_clean")

    cfg = CleanConfig(
        label_col="label",
        day_col="day",
        min_non_null_frac=0.40,
        debug=True
    )

    build_clients(BASE_DIR, OUT_DIR, cfg)


