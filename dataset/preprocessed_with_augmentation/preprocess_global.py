# preprocess_global.py
from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass,field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from extract_ts_features import extract_ts_features, TSFeatureConfig, infer_ts_columns
from ts_augment import TSAugmentConfig, augment_ts_dataframe


# -----------------------------
# Config
# -----------------------------
@dataclass
class CleanConfigGlobal:
    label_col: str = "label"
    day_col: Optional[str] = "day"

    # train vs infer
    mode: str = "train"  # "train" | "infer"
    drop_label_zero: bool = True
    min_non_null_frac: float = 0.40

    # IQR
    iqr_k: float = 1.5

    # TS
    use_ts_features: bool = True
    ts_drop_original_cols: bool = True
    ts_drop_negative_values: bool = True
    ts_add_quality_features: bool = True
    ts_max_neg_frac_raw: float = 0.50
    ts_min_valid_points: int = 5

    # TS augmentation
    ts_augment: bool = False
    ts_aug_cfg: TSAugmentConfig = field(default_factory=TSAugmentConfig)

    # debug
    debug: bool = True


@dataclass
class GlobalStats:
    # mediana globale per colonna
    medians: Dict[str, float]
    # Q1/Q3 globali per colonna
    q1: Dict[str, float]
    q3: Dict[str, float]

    def to_json(self) -> str:
        return json.dumps({"medians": self.medians, "q1": self.q1, "q3": self.q3})

    @staticmethod
    def from_json(s: str) -> "GlobalStats":
        obj = json.loads(s)
        return GlobalStats(medians=obj["medians"], q1=obj["q1"], q3=obj["q3"])


# -----------------------------
# Selettori feature: coerenti col tuo codice attuale
# -----------------------------
def _select_numeric_feature_cols(df: pd.DataFrame, cfg: CleanConfigGlobal) -> List[str]:
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _select_numeric_original_cols_for_low_info(df: pd.DataFrame, cfg: CleanConfigGlobal) -> List[str]:
    cols = _select_numeric_feature_cols(df, cfg)
    return [c for c in cols if not (c.startswith("ts__") or c.endswith("__clean") or c.endswith("__is_outlier"))]


def _coerce_numeric_features(df: pd.DataFrame, cfg: CleanConfigGlobal) -> pd.DataFrame:
    out = df.copy()
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)
    for c in out.columns:
        if c not in exclude:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# -----------------------------
# Pulizia righe (come la tua)
# -----------------------------
def drop_invalid_labels(df: pd.DataFrame, cfg: CleanConfigGlobal) -> pd.DataFrame:
    if cfg.label_col not in df.columns:
        return df
    out = df.dropna(subset=[cfg.label_col]).copy()
    if cfg.drop_label_zero:
        out = out[out[cfg.label_col] != 0]
    return out


def drop_low_info_days(df: pd.DataFrame, cfg: CleanConfigGlobal) -> pd.DataFrame:
    feat = _select_numeric_original_cols_for_low_info(df, cfg)
    if not feat:
        return df
    frac = df[feat].notna().mean(axis=1)
    return df[frac >= cfg.min_non_null_frac].copy()


# -----------------------------
# GLOBAL median imputation
# -----------------------------
def impute_missing_values_global(df: pd.DataFrame, cfg: CleanConfigGlobal, gs: GlobalStats) -> pd.DataFrame:
    out = df.copy()
    feat = _select_numeric_original_cols_for_low_info(out, cfg)
    if not feat:
        return out

    fill_map = {c: float(gs.medians.get(c, 0.0)) for c in feat}
    out[feat] = out[feat].fillna(value=fill_map)
    return out


# -----------------------------
# GLOBAL IQR outlier handling
# (stessa logica del tuo: crea __is_outlier e __clean, poi finalize sovrascrive)
# -----------------------------
def handle_outliers_iqr_global(df: pd.DataFrame, cfg: CleanConfigGlobal, gs: GlobalStats) -> pd.DataFrame:
    out = df.copy()
    feat = _select_numeric_original_cols_for_low_info(out, cfg)

    for c in feat:
        s = out[c]
        out[f"{c}__is_outlier"] = False
        out[f"{c}__clean"] = s

        q1 = gs.q1.get(c, np.nan)
        q3 = gs.q3.get(c, np.nan)
        if not np.isfinite(q1) or not np.isfinite(q3):
            continue

        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue

        lo, hi = q1 - cfg.iqr_k * iqr, q3 + cfg.iqr_k * iqr
        mask = (s < lo) | (s > hi)
        out[f"{c}__is_outlier"] = mask.fillna(False)

        # replacement: mediana globale (coerente e stabile)
        repl = float(gs.medians.get(c, np.nan))
        if not np.isfinite(repl):
            repl = float(np.nanmedian(s.values)) if np.isfinite(np.nanmedian(s.values)) else 0.0

        out[f"{c}__clean"] = s.where(~mask, repl)

    return out


def fill_remaining_nans_global(df: pd.DataFrame, cfg: CleanConfigGlobal, gs: GlobalStats) -> pd.DataFrame:
    """
    Safety: riempi NaN residui sulle numeriche con mediana globale (fallback 0).
    """
    out = df.copy()
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)

    num_cols = [
        c for c in out.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(out[c])
        and not c.endswith("__is_outlier")
    ]
    if not num_cols:
        return out

    fill_map = {c: float(gs.medians.get(c, 0.0)) for c in num_cols}
    out[num_cols] = out[num_cols].fillna(value=fill_map)
    return out


# -----------------------------
# Drop colonne: IDENTICO al tuo set esplicito
# -----------------------------
_COLS_TO_DROP_EXPLICIT = {
    "ts__resp_time_series__nan_frac_raw",
    "ts__stress_time_series__nan_frac_raw",
    "ts__hr_time_series__neg_frac_raw",
    "ts__hr_time_series__neg_count_raw",
    "act_activeTime",
    "ts__hr_time_series__used",
}


def finalize_clean_columns(df: pd.DataFrame, cfg: CleanConfigGlobal) -> pd.DataFrame:
    out = df.copy()
    orig_cols = _select_numeric_original_cols_for_low_info(out, cfg)

    # sovrascrivi le colonne originali con la versione pulita (come facevi prima)
    for c in orig_cols:
        clean_c = f"{c}__clean"
        if clean_c in out.columns:
            out[c] = out[clean_c]

    drop_cols = [
        c for c in out.columns
        if c.endswith("__is_outlier") or c.endswith("__clean") or c in _COLS_TO_DROP_EXPLICIT
    ]
    return out.drop(columns=drop_cols, errors="ignore")


# -----------------------------
# GLOBAL STATS: calcolo (centralizzato/offline)
# -----------------------------
def compute_global_stats_from_csvs(
    csv_paths: List[str],
    cfg: CleanConfigGlobal,
) -> GlobalStats:
    """
    Calcola mediana globale + Q1/Q3 globali per colonna, leggendo i csv (puoi passare tutti i file dei 9 client).
    Nota: calcolo esatto (centralizzato) -> semplice e stabile con i tuoi volumi.
    """
    frames: List[pd.DataFrame] = []
    for p in csv_paths:
        df = pd.read_csv(p, sep=";").drop(columns=["Unnamed: 0"], errors="ignore")
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # (opzionale) TS features prima di stats, così le stats includono anche ts__* (se vuoi)
    # Nel tuo selettore "original cols" esclude ts__, quindi qui calcoliamo stats solo sulle original.
    if cfg.day_col and cfg.day_col in all_df.columns:
        all_df[cfg.day_col] = pd.to_numeric(all_df[cfg.day_col], errors="coerce")

    all_df = _coerce_numeric_features(all_df, cfg)
    feat = _select_numeric_original_cols_for_low_info(all_df, cfg)
    if not feat:
        return GlobalStats(medians={}, q1={}, q3={})

    med = all_df[feat].median(numeric_only=True).fillna(0.0).to_dict()
    q1 = all_df[feat].quantile(0.25, numeric_only=True).fillna(np.nan).to_dict()
    q3 = all_df[feat].quantile(0.75, numeric_only=True).fillna(np.nan).to_dict()

    # cast a float semplice
    med = {k: float(v) for k, v in med.items()}
    q1 = {k: float(v) if np.isfinite(v) else float("nan") for k, v in q1.items()}
    q3 = {k: float(v) if np.isfinite(v) else float("nan") for k, v in q3.items()}

    return GlobalStats(medians=med, q1=q1, q3=q3)


# -----------------------------
# Pipeline singolo user/day DF (come la tua, ma con GLOBAL stats + TS aug)
# -----------------------------
def clean_user_df_global(df: pd.DataFrame, cfg: CleanConfigGlobal, gs: GlobalStats) -> pd.DataFrame:
    out = df.copy()
    n_rows_start = len(out)

    # 1) TS augmentation (sulle colonne TS originali, PRIMA dell'extract)
    if cfg.ts_augment:
        ts_cols = infer_ts_columns(out, TSFeatureConfig(ts_cols=None))
        out = augment_ts_dataframe(out, ts_cols=ts_cols, cfg=cfg.ts_aug_cfg)

    # 2) TS features (come tuo flusso)
    if cfg.use_ts_features:
        ts_cfg = TSFeatureConfig(
            ts_cols=None,
            drop_original_ts_cols=cfg.ts_drop_original_cols,
            drop_negative_values=cfg.ts_drop_negative_values,
            add_quality_features=cfg.ts_add_quality_features,
            max_neg_frac_raw=cfg.ts_max_neg_frac_raw,
            min_valid_points=cfg.ts_min_valid_points,
        )
        out = extract_ts_features(out, ts_cfg)

    # 3) conversioni numeriche
    if cfg.day_col and cfg.day_col in out.columns:
        out[cfg.day_col] = pd.to_numeric(out[cfg.day_col], errors="coerce")
    out = _coerce_numeric_features(out, cfg)

    # 4) drop righe solo in train (come prima)
    if cfg.mode == "train":
        out = drop_invalid_labels(out, cfg)
        out = drop_low_info_days(out, cfg)

    # 5) imputazione NaN con mediana GLOBALE
    out = impute_missing_values_global(out, cfg, gs)

    # 6) outlier IQR con Q1/Q3 GLOBALI
    out = handle_outliers_iqr_global(out, cfg, gs)

    # 7) safety fill NaN residui con mediana globale
    out = fill_remaining_nans_global(out, cfg, gs)

    # 8) drop colonne (incluse quelle esplicite identiche)
    out = finalize_clean_columns(out, cfg)

    if cfg.debug:
        n_nan = int(out.isna().sum().sum())
        print(f"Rows: {n_rows_start} -> {len(out)} | Cols: {out.shape[1]} | Total NaN: {n_nan}")

    return out.reset_index(drop=True)


# -----------------------------
# Helper: build clients come prima (ma serve passare global stats)
# -----------------------------
def read_user_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";").drop(columns=["Unnamed: 0"], errors="ignore")


def parse_user_id(filename: str) -> str:
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("_")
    return parts[parts.index("user") + 1] if "user" in parts else base


def build_clients_with_global_stats(base_dir: str, out_dir: str, cfg: CleanConfigGlobal, gs: GlobalStats):
    os.makedirs(out_dir, exist_ok=True)
    group_dirs = sorted(d for d in glob.glob(os.path.join(base_dir, "group*")) if os.path.isdir(d))

    print(f"\nClient trovati: {len(group_dirs)}\n")

    for gdir in group_dirs:
        client_id = os.path.basename(gdir)
        user_files = sorted(glob.glob(os.path.join(gdir, "*.csv")))
        print(f"Client {client_id} | utenti: {len(user_files)}")

        dfs = []
        for p in user_files:
            df = read_user_csv(p)
            df["client_id"] = client_id
            df["user_id"] = parse_user_id(p)
            df["source_file"] = os.path.basename(p)

            df = clean_user_df_global(df, cfg, gs)
            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)
        if cfg.day_col and cfg.day_col in merged.columns:
            merged = merged.sort_values([cfg.day_col, "user_id"])

        out_path = os.path.join(out_dir, f"{client_id}_merged_clean.csv")
        merged.to_csv(out_path, index=False)
        print(f"OK salvato {client_id}: {merged.shape[0]} righe | {merged.shape[1]} colonne")

def build_x_test_with_global_stats(
    x_test_path: str,
    out_path: str,
    cfg_train: CleanConfigGlobal,
    gs: GlobalStats,
):
    """
    Pulisce x_test usando le stesse GlobalStats (mediana + Q1/Q3) calcolate sul train.
    Non deve fare filtri "da training" che buttano righe a caso.
    """
    df = pd.read_csv(x_test_path, sep=";").drop(columns=["Unnamed: 0"], errors="ignore")

    # Coerenza con il vecchio script
    df["client_id"] = "test"
    df["user_id"] = "test"
    df["source_file"] = os.path.basename(x_test_path)

    # Config inferenza: niente drop per label==0 ecc.
    cfg_test = CleanConfigGlobal(**{**cfg_train.__dict__, "mode": "infer", "debug": cfg_train.debug})

    clean = clean_user_df_global(df, cfg_test, gs)
    clean.to_csv(out_path, index=False)

    print(f"✔ SALVATO X_TEST: {clean.shape[0]} righe | {clean.shape[1]} colonne -> {out_path}")



if __name__ == "__main__":
    import os
    import glob

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Raw dataset (come prima)
    TRAIN_BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../raw_dataset"))

    # >>> OUTPUT coerenti con get_train_path/get_test_path (DATASET="new")
    OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "clients_dataset"))
    X_TEST_OUT = os.path.abspath(os.path.join(SCRIPT_DIR, "x_test_clean.csv"))

    os.makedirs(OUT_DIR, exist_ok=True)

    all_csvs = sorted(glob.glob(os.path.join(TRAIN_BASE_DIR, "group*", "*.csv")))
    cfg = CleanConfigGlobal(mode="train", debug=True)

    gs = compute_global_stats_from_csvs(all_csvs, cfg)
    print("GlobalStats calcolate:", len(gs.medians), "colonne")

    # 2) costruzione dataset client con preprocessing GLOBAL
    build_clients_with_global_stats(TRAIN_BASE_DIR, OUT_DIR, cfg, gs)

    # 3) salva stats
    with open(os.path.join(OUT_DIR, "global_stats.json"), "w", encoding="utf-8") as f:
        f.write(gs.to_json())

    # --- X_TEST (come nel vecchio script) ---
    X_TEST_PATH = os.path.abspath(os.path.join(TRAIN_BASE_DIR, "x_test.csv"))
    # Se invece x_test.csv sta fuori da TRAIN_BASE_DIR, allora usa:
    # X_TEST_PATH = os.path.abspath(os.path.join(TRAIN_BASE_DIR, "..", "x_test.csv"))

    if os.path.exists(X_TEST_PATH):
        build_x_test_with_global_stats(X_TEST_PATH, X_TEST_OUT, cfg, gs)
    else:
        print(f"⚠️ x_test.csv non trovato in: {X_TEST_PATH}")
