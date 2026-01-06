# preprocess_global.py
from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    min_non_null_frac: float = 0.8

    # IQR (usato SOLO per clipping)
    iqr_k: float = 1.5

    # Domain sanity checks (wearable)
    # HR
    hr_min: float = 30.0
    hr_max: float = 220.0
    # Respiration
    resp_min: float = 5.0
    resp_max: float = 60.0
    # Sleep/seconds bounds
    seconds_min: float = 0.0
    seconds_max: float = 86400.0  # 24h
    # Stress bounds (se vuoi disattivare, metti None)
    stress_min: float = 0.0
    stress_max: float = 100.0

    # TS
    use_ts_features: bool = True
    ts_drop_original_cols: bool = True
    ts_drop_negative_values: bool = True
    ts_add_quality_features: bool = True
    ts_max_neg_frac_raw: float = 0.50
    ts_min_valid_points: int = 5

    # TS augmentation
    ts_augment: bool = True
    ts_aug_cfg: TSAugmentConfig = field(default_factory=TSAugmentConfig)

    # debug
    debug: bool = True


@dataclass
class GlobalStats:
    # mediana globale per colonna (resta per compatibilità / json)
    medians: Dict[str, float]
    # Q1/Q3 globali per colonna (per clipping IQR)
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


def _is_quality_or_flag_col(c: str) -> bool:
    """
    Colonne che NON vanno toccate da outlier clipping/imputation aggressivi.
    (Non cambia la tua lista drop: serve solo a evitare trattamenti non sensati.)
    """
    if c.startswith("ts__"):
        if c.endswith("__used"):
            return True
        if "__len" in c or "__len_raw" in c:
            return True
        if "__nan_frac" in c:
            return True
        if "__neg_frac" in c or "__neg_count" in c:
            return True
    if c.endswith("__missing") or c.endswith("__is_outlier") or c.endswith("__clean"):
        return True
    return False

def _should_add_missing_flag(c: str) -> bool:
    """
    Per evitare di raddoppiare le colonne:
    - missing flag SOLO per feature "raw" scalari (hr_/resp_/sleep_/act_/str_)
    - NON per statistiche ts__... (per quelle usiamo used/len come qualità)
    """
    return c.startswith(("hr_", "resp_", "sleep_", "act_", "str_"))



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
# Domain sanity checks (valori impossibili -> NaN)
# -----------------------------
def apply_domain_sanity_checks(df: pd.DataFrame, cfg: CleanConfigGlobal) -> pd.DataFrame:
    """
    Regole di dominio per wearable:
    - Non sono outlier statistici: sono valori improbabili/impossibili.
    - Li trasformiamo in NaN (poi verranno gestiti da missing flags + fill 0).
    """
    out = df.copy()

    # HR scalari: hr_* e sleep_avgHeartRate
    hr_cols = [c for c in out.columns if c.startswith("hr_")] + (["sleep_avgHeartRate"] if "sleep_avgHeartRate" in out.columns else [])
    for c in hr_cols:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            out[c] = s.mask((s < cfg.hr_min) | (s > cfg.hr_max), np.nan)

    # Respirazione: resp_* e sleep_*Respiration*
    resp_cols = [c for c in out.columns if c.startswith("resp_")] + [c for c in out.columns if "Respiration" in c]
    for c in resp_cols:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            out[c] = s.mask((s < cfg.resp_min) | (s > cfg.resp_max), np.nan)

    # Secondi (sonno): *Seconds
    sec_cols = [c for c in out.columns if c.endswith("Seconds")]
    for c in sec_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = s.mask((s < cfg.seconds_min) | (s > cfg.seconds_max), np.nan)

    # Stress (se nel tuo dataset è 0..100; se non vuoi, commenta)
    stress_cols = [c for c in out.columns if c.startswith("str_")] + (["sleep_avgSleepStress"] if "sleep_avgSleepStress" in out.columns else [])
    for c in stress_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = s.mask((s < cfg.stress_min) | (s > cfg.stress_max), np.nan)

    # Attività: non negativa (se presenti)
    for c in ("act_totalCalories", "act_activeKilocalories", "act_distance", "act_activeTime"):
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            out[c] = s.mask(s < 0, np.nan)

    return out


# -----------------------------
# Imputazione: 0 + missing flags (NO mediana globale)
# -----------------------------
#def impute_missing_values_global(df: pd.DataFrame, cfg: CleanConfigGlobal, gs: GlobalStats) -> pd.DataFrame:
    """
    Per NN/MLP:
    - crea flag <col>__missing (0/1) sulle feature "original" (non ts__...)
    - fill NaN con 0.0 (valore neutro)
    Non usa mediane globali.
    """
    out = df.copy()
    feat = _select_numeric_original_cols_for_low_info(out, cfg)
    if not feat:
        return out

    feat = [c for c in feat if not _is_quality_or_flag_col(c)]
    for c in feat:
        if _should_add_missing_flag(c):
            out[f"{c}__missing"] = out[c].isna().astype(np.int8)

    out[feat] = out[feat].fillna(0.0)
    return out

#con mediana
def impute_missing_values_global(df: pd.DataFrame, cfg: CleanConfigGlobal, gs: GlobalStats) -> pd.DataFrame:
    """
    - crea flag <col>__missing (0/1) sulle feature "original" (non ts__...)
    - imputazione NaN con mediana globale per colonna (fallback 0.0)
    """
    out = df.copy()
    feat = _select_numeric_original_cols_for_low_info(out, cfg)
    if not feat:
        return out

    feat = [c for c in feat if not _is_quality_or_flag_col(c)]

    # missing flags (come prima)
    for c in feat:
        if _should_add_missing_flag(c):
            out[f"{c}__missing"] = out[c].isna().astype(np.int8)

    # imputazione con mediana globale colonna-per-colonna
    for c in feat:
        med = gs.medians.get(c, 0.0)
        if not np.isfinite(med):
            med = 0.0
        out[c] = out[c].fillna(med)

    return out



# -----------------------------
# Outlier IQR: CLIPPING (NO mediana globale)
# -----------------------------
def handle_outliers_iqr_global(df: pd.DataFrame, cfg: CleanConfigGlobal, gs: GlobalStats) -> pd.DataFrame:
    """
    Stessa struttura della tua:
    - crea __is_outlier e __clean
    - finalize sovrascrive gli originali con __clean
    Ma invece di rimpiazzare con mediana, fa clipping ai fence IQR.
    """
    out = df.copy()
    feat = _select_numeric_original_cols_for_low_info(out, cfg)
    feat = [c for c in feat if not _is_quality_or_flag_col(c)]

    for c in feat:
        s = pd.to_numeric(out[c], errors="coerce")
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

        # CLIPPING invece di replacement con mediana
        out[f"{c}__clean"] = s.clip(lower=lo, upper=hi)

    return out





#def fill_remaining_nans_global(df: pd.DataFrame, cfg: CleanConfigGlobal, gs: GlobalStats) -> pd.DataFrame:
    """
    Safety finale:
    - crea missing flags per numeriche rimaste NaN (se non già create)
    - fillna(0.0) per tutte le numeriche (escludendo label/id/day)
    Non usa mediane globali.
    """
    out = df.copy()
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)

    num_cols = [
        c for c in out.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(out[c])
        and not c.endswith("__is_outlier")
        and not c.endswith("__clean")
    ]
    if not num_cols:
        return out

    # crea missing flags dove mancano (solo per non-quality)
    for c in num_cols:
        if _is_quality_or_flag_col(c):
            continue
        if not _should_add_missing_flag(c):
            continue
        miss_c = f"{c}__missing"
        if miss_c not in out.columns:
            out[miss_c] = out[c].isna().astype(np.int8)

    out[num_cols] = out[num_cols].fillna(0.0)
    return out

#con mediana
def fill_remaining_nans_global(df: pd.DataFrame, cfg: CleanConfigGlobal, gs: GlobalStats) -> pd.DataFrame:
    """
    OPZIONE: MEDIANA globale per colonna (fallback 0.0)
    - crea missing flags per numeriche rimaste NaN (se non già create)
    - imputazione NaN con mediana globale
    """
    out = df.copy()
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)

    num_cols = [
        c for c in out.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(out[c])
        and not c.endswith("__is_outlier")
        and not c.endswith("__clean")
    ]
    if not num_cols:
        return out

    # missing flags dove mancano (solo per wearable scalari come da tua logica)
    for c in num_cols:
        if _is_quality_or_flag_col(c):
            continue
        if not _should_add_missing_flag(c):
            continue
        miss_c = f"{c}__missing"
        if miss_c not in out.columns:
            out[miss_c] = out[c].isna().astype(np.int8)

    # imputazione: MEDIANA
    for c in num_cols:
        med = gs.medians.get(c, 0.0)
        if not np.isfinite(med):
            med = 0.0
        out[c] = out[c].fillna(med)

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

    # sovrascrivi le colonne originali con la versione pulita
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
    frames: List[pd.DataFrame] = []
    for p in csv_paths:
        df = pd.read_csv(p, sep=";").drop(columns=["Unnamed: 0"], errors="ignore")
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # Estrai TS feature anche qui (come prima)
    if cfg.use_ts_features:
        ts_cfg = TSFeatureConfig(
            ts_cols=None,
            drop_original_ts_cols=cfg.ts_drop_original_cols,
            drop_negative_values=cfg.ts_drop_negative_values,
            add_quality_features=cfg.ts_add_quality_features,
            max_neg_frac_raw=cfg.ts_max_neg_frac_raw,
            min_valid_points=cfg.ts_min_valid_points,
        )
        all_df = extract_ts_features(all_df, ts_cfg)

    # coercion
    if cfg.day_col and cfg.day_col in all_df.columns:
        all_df[cfg.day_col] = pd.to_numeric(all_df[cfg.day_col], errors="coerce")
    all_df = _coerce_numeric_features(all_df, cfg)

    # applichiamo anche qui i sanity checks, così Q1/Q3 non vengono “sporcati” da valori impossibili
    all_df = apply_domain_sanity_checks(all_df, cfg)

    feat = _select_numeric_feature_cols(all_df, cfg)
    feat = [c for c in feat if not c.endswith("__is_outlier") and not c.endswith("__clean")]
    if not feat:
        return GlobalStats(medians={}, q1={}, q3={})

    # Le mediane restano per compatibilità, anche se non le userai più per imputare.
    med = all_df[feat].median(numeric_only=True).fillna(0.0).to_dict()
    q1 = all_df[feat].quantile(0.25, numeric_only=True).fillna(np.nan).to_dict()
    q3 = all_df[feat].quantile(0.75, numeric_only=True).fillna(np.nan).to_dict()

    med = {k: float(v) for k, v in med.items()}
    q1 = {k: float(v) if np.isfinite(v) else float("nan") for k, v in q1.items()}
    q3 = {k: float(v) if np.isfinite(v) else float("nan") for k, v in q3.items()}

    return GlobalStats(medians=med, q1=q1, q3=q3)


# -----------------------------
# Pipeline singolo user/day DF (come la tua, ma corretto)
# -----------------------------
def clean_user_df_global(df: pd.DataFrame, cfg: CleanConfigGlobal, gs: GlobalStats) -> pd.DataFrame:
    out = df.copy()
    n_rows_start = len(out)

    # 1) TS augmentation (prima dell'extract)
    if cfg.ts_augment:
        ts_cols = infer_ts_columns(out, TSFeatureConfig(ts_cols=None))
        out = augment_ts_dataframe(out, ts_cols=ts_cols, cfg=cfg.ts_aug_cfg)

    # 2) TS features
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

    # Proxy "used" per HR TS (dato che ts__hr_time_series__used viene droppata esplicitamente)
    if "ts__hr_time_series__len" in out.columns:
        out["ts__hr_time_series__used_proxy"] = (
                pd.to_numeric(out["ts__hr_time_series__len"], errors="coerce").fillna(0) >= cfg.ts_min_valid_points
        ).astype(np.int8)
    else:
        out["ts__hr_time_series__used_proxy"] = 0

    # 3.5) sanity checks wearable
    out = apply_domain_sanity_checks(out, cfg)

    # 4) drop righe solo in train (come prima)
    if cfg.mode == "train":
        out = drop_invalid_labels(out, cfg)
        out = drop_low_info_days(out, cfg)

    # 5) imputazione: 0 + missing flags (NO mediana)
    out = impute_missing_values_global(out, cfg, gs)

    # 6) outlier IQR: clipping (NO mediana)
    out = handle_outliers_iqr_global(out, cfg, gs)

    if cfg.debug:
        print_nan_report(out, title="PRE fill_remaining_nans_global")

    # 7) safety: flags + fill 0 (NO mediana)
    out = fill_remaining_nans_global(out, cfg, gs)

    if cfg.debug:
        print_nan_report(out, title="POST fill_remaining_nans_global")


    # 8) drop colonne (identico)
    out = finalize_clean_columns(out, cfg)

    if cfg.debug:
        n_nan = int(out.isna().sum().sum())
        print(f"Rows: {n_rows_start} -> {len(out)} | Cols: {out.shape[1]} | Total NaN: {n_nan}")

    return out.reset_index(drop=True)


# -----------------------------
# Helper: build clients come prima
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
        print_nan_report(merged, title=f"{client_id} (merged clean)")

        if cfg.day_col and cfg.day_col in merged.columns:
            merged = merged.sort_values([cfg.day_col, "user_id"])
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
    df = pd.read_csv(x_test_path, sep=";").drop(columns=["Unnamed: 0"], errors="ignore")
    df["client_id"] = "test"
    df["user_id"] = "test"
    df["source_file"] = os.path.basename(x_test_path)

    cfg_test = CleanConfigGlobal(**{**cfg_train.__dict__, "mode": "infer", "debug": cfg_train.debug})

    clean = clean_user_df_global(df, cfg_test, gs)
    clean.to_csv(out_path, index=False)
    print(f"✔ SALVATO X_TEST: {clean.shape[0]} righe | {clean.shape[1]} colonne -> {out_path}")



def print_nan_report(df: pd.DataFrame, title: str) -> None:
    total_nans = int(df.isna().sum().sum())
    rows_with_nan = int(df.isna().any(axis=1).sum())
    n_rows = int(len(df))

    if n_rows > 0:
        per_row = df.isna().sum(axis=1)
        mn = int(per_row.min())
        av = float(per_row.mean())
        md = float(per_row.median())
        mx = int(per_row.max())
    else:
        mn = mx = 0
        av = md = 0.0

    print(
        f"[{title}] Rows={n_rows} | Total NaN={total_nans} | Rows w/ >=1 NaN={rows_with_nan} "
        f"| NaN/row min={mn} mean={av:.3f} median={md:.1f} max={mx}"
    )



if __name__ == "__main__":
    import os
    import glob

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    TRAIN_BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../raw_dataset"))
    OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "clients_dataset"))
    X_TEST_OUT = os.path.abspath(os.path.join(SCRIPT_DIR, "x_test_clean.csv"))

    os.makedirs(OUT_DIR, exist_ok=True)

    all_csvs = sorted(glob.glob(os.path.join(TRAIN_BASE_DIR, "group*", "*.csv")))
    cfg = CleanConfigGlobal(mode="train", debug=True)

    gs = compute_global_stats_from_csvs(all_csvs, cfg)
    print("GlobalStats calcolate:", len(gs.medians), "colonne")

    build_clients_with_global_stats(TRAIN_BASE_DIR, OUT_DIR, cfg, gs)

    with open(os.path.join(OUT_DIR, "global_stats.json"), "w", encoding="utf-8") as f:
        f.write(gs.to_json())

    X_TEST_PATH = os.path.abspath(os.path.join(TRAIN_BASE_DIR, "x_test.csv"))
    if os.path.exists(X_TEST_PATH):
        build_x_test_with_global_stats(X_TEST_PATH, X_TEST_OUT, cfg, gs)
    else:
        print(f"⚠️ x_test.csv non trovato in: {X_TEST_PATH}")
