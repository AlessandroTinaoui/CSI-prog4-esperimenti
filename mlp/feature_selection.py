#!/usr/bin/env python3
"""
Feature selection offline for your pre-processed (tabular) datasets.

What it does (non-supervised, FL-safe):
1) Load all client CSVs (already preprocessed).
2) Compute missing-rate per feature (BEFORE imputing).
3) Impute with median (for stats only), compute variance, drop near-constant.
4) Apply simple semantic pruning for time-series "quality" features:
   - keep only __used and __nan_frac_raw among quality columns by default
5) Drop one feature from pairs/groups with high feature-feature correlation.
6) Save selected feature list to JSON.
7) Optionally write reduced CSVs (keeping label + optional ID/meta columns).

Usage examples:
  python feature_selection_offline.py \
      --data-dir mlp/client/clients_data \
      --pattern "group*_merged_clean.csv" \
      --out-dir mlp/results \
      --write-csv --csv-suffix "_fs"

Notes:
- This script does NOT use the target label for selection (recommended for FL).
- If your CSVs have no NaNs because you already filled them, missing-rate will be ~0.
  In that case, missing-rate filtering won't remove anything (which is fine).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dataset.dataset_cfg import get_train_path

DEFAULT_DROP_COLS = ["day", "client_id", "user_id", "source_file"]
DEFAULT_LABEL_COL = "label"


def is_quality_feature(col: str) -> bool:
    """Heuristic: time-series quality/availability metadata."""
    return (
        "__used" in col
        or "__len_raw" in col
        or "__nan_frac" in col
        or "__neg_frac" in col
        or "__too_short" in col
    )


def should_keep_quality_feature(col: str) -> bool:
    """
    Keep only the most informative quality features by default.
    Tune this if you want to keep more/less.
    """
    return col.endswith("__used") or col.endswith("__nan_frac_raw")


def _read_one_csv(
    path: Path,
    label_col: str,
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series | None]:
    df = pd.read_csv(path)

    y = None
    if label_col in df.columns:
        y = df[label_col].copy()

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if label_col in df.columns:
        cols_to_drop.append(label_col)

    X = df.drop(columns=cols_to_drop, errors="ignore").copy()

    # Force numeric (coerce non-numeric -> NaN)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X, y


def load_global_X(
    data_dir: Path,
    pattern: str,
    label_col: str,
    drop_cols: List[str],
    max_files: int | None = None,
) -> pd.DataFrame:
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {data_dir}/{pattern}")

    if max_files is not None:
        paths = paths[: max_files]

    dfs = []
    for p in paths:
        X, _ = _read_one_csv(p, label_col=label_col, drop_cols=drop_cols)
        dfs.append(X)

    # Align columns (union) across clients
    X_all = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    return X_all


def filter_by_missing_rate(
    X: pd.DataFrame,
    max_missing_rate: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    missing_rate = X.isna().mean().to_dict()
    keep = [c for c, r in missing_rate.items() if float(r) <= max_missing_rate]
    return X[keep], missing_rate


def filter_by_variance(
    X: pd.DataFrame,
    min_variance: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # For variance computation, fill NaNs with per-column median (robust)
    med = X.median(numeric_only=True)
    X_filled = X.fillna(med)
    var = X_filled.var(numeric_only=True).to_dict()
    keep = [c for c, v in var.items() if float(v) > min_variance]
    return X[keep], var


def prune_quality_features(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Keep all value features. For quality features, keep only selected subset.
    Returns pruned X + map explaining drops.
    """
    reason: Dict[str, str] = {}
    cols = list(X.columns)

    quality = [c for c in cols if is_quality_feature(c)]
    keep_quality = [c for c in quality if should_keep_quality_feature(c)]
    value = [c for c in cols if c not in quality]

    # Mark dropped quality features
    drop_quality = [c for c in quality if c not in keep_quality]
    for c in drop_quality:
        reason[c] = "dropped_quality_feature"

    keep = value + keep_quality
    return X[keep], reason


def drop_highly_correlated(
    X: pd.DataFrame,
    corr_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Drops one feature from any pair with abs(corr) > threshold.
    Uses median-filled data to compute correlation.
    """
    med = X.median(numeric_only=True)
    X_filled = X.fillna(med)

    # If only 0-1 columns, nothing to do
    if X_filled.shape[1] <= 1:
        return X, {}

    corr = X_filled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = []
    reason: Dict[str, str] = {}

    for col in upper.columns:
        if (upper[col] > corr_threshold).any():
            to_drop.append(col)
            reason[col] = "high_correlation"

    if not to_drop:
        return X, {}

    X2 = X.drop(columns=to_drop, errors="ignore")
    return X2, reason


def write_selected_csvs(
    data_dir: Path,
    pattern: str,
    out_dir: Path,
    selected_features: List[str],
    label_col: str,
    keep_meta_cols: List[str],
    suffix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(data_dir.glob(pattern))
    for p in paths:
        df = pd.read_csv(p)

        cols = []
        # Keep meta cols if present
        for c in keep_meta_cols:
            if c in df.columns:
                cols.append(c)

        # Keep selected features if present
        for c in selected_features:
            if c in df.columns:
                cols.append(c)

        # Keep label if present
        if label_col in df.columns:
            cols.append(label_col)

        df_out = df[cols].copy()

        # Write next to original (or into out_dir) with suffix
        out_path = out_dir / f"{p.stem}{suffix}{p.suffix}"
        df_out.to_csv(out_path, index=False)

BASE_PROJ = Path(__file__).resolve().parents[1]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="../"+get_train_path(), help="Directory containing client CSVs")
    ap.add_argument("--pattern", type=str, default="group*_merged_clean.csv", help="Glob pattern for CSVs")
    ap.add_argument("--out-dir", type=str, default="../results", help="Output directory for JSON and (optional) CSVs")

    ap.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL, help="Target column name")
    ap.add_argument(
        "--drop-cols",
        type=str,
        default=",".join(DEFAULT_DROP_COLS),
        help="Comma-separated columns to drop (meta columns)",
    )

    ap.add_argument("--max-missing-rate", type=float, default=0.50, help="Drop features with missing rate above this")
    ap.add_argument("--min-variance", type=float, default=1e-6, help="Drop features with variance <= this")
    ap.add_argument("--corr-threshold", type=float, default=0.95, help="Drop features with abs(corr) > this")

    ap.add_argument(
        "--write-csv",
        action="store_true",
        help="If set, write reduced CSVs containing only selected features",
    )
    ap.add_argument(
        "--csv-out-dir",
        type=str,
        default="",
        help="Where to write reduced CSVs. Default: same as --out-dir",
    )
    ap.add_argument("--csv-suffix", type=str, default="_fs", help="Suffix for reduced CSV filenames")

    ap.add_argument(
        "--keep-meta-cols",
        type=str,
        default="day,client_id,user_id,source_file",
        help="Comma-separated meta columns to keep in output CSVs (if present)",
    )
    ap.add_argument("--max-files", type=int, default=0, help="Load only first N files (0 = all)")

    args = ap.parse_args()

    data_dir = BASE_PROJ/get_train_path()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    keep_meta_cols = [c.strip() for c in args.keep_meta_cols.split(",") if c.strip()]

    max_files = None if args.max_files == 0 else int(args.max_files)

    # 1) Load global X (union of columns)
    X = load_global_X(
        data_dir=data_dir,
        pattern=args.pattern,
        label_col=args.label_col,
        drop_cols=drop_cols,
        max_files=max_files,
    )

    report: Dict[str, object] = {
        "n_rows": int(X.shape[0]),
        "n_cols_initial": int(X.shape[1]),
        "pattern": args.pattern,
        "thresholds": {
            "max_missing_rate": float(args.max_missing_rate),
            "min_variance": float(args.min_variance),
            "corr_threshold": float(args.corr_threshold),
        },
    }

    # 2) Missing-rate filter (pre-imputation)
    X, missing_rate = filter_by_missing_rate(X, max_missing_rate=float(args.max_missing_rate))
    report["n_cols_after_missing"] = int(X.shape[1])

    # 3) Variance filter (median-imputed for stats only)
    X, variance = filter_by_variance(X, min_variance=float(args.min_variance))
    report["n_cols_after_variance"] = int(X.shape[1])

    # 4) Quality feature pruning (semantic)
    X, quality_drop_reason = prune_quality_features(X)
    report["n_cols_after_quality_prune"] = int(X.shape[1])

    # 5) Correlation filter
    X, corr_drop_reason = drop_highly_correlated(X, corr_threshold=float(args.corr_threshold))
    report["n_cols_after_corr"] = int(X.shape[1])

    selected_features = sorted(list(X.columns))

    # Save outputs
    features_path = out_dir / "selected_features.json"
    features_path.write_text(json.dumps(selected_features, indent=2), encoding="utf-8")

    # Save an audit/report JSON (useful for your report)
    audit = {
        "summary": report,
        "dropped": {
            "quality": quality_drop_reason,
            "correlation": corr_drop_reason,
        },
        "stats": {
            "missing_rate": missing_rate,
            "variance_after_median_impute": variance,
        },
        "selected_features": selected_features,
    }
    audit_path = out_dir / "feature_selection_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"Loaded rows: {X.shape[0]}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Wrote: {features_path}")
    print(f"Wrote: {audit_path}")

    # Optionally write reduced CSVs
    if args.write_csv:
        csv_out_dir = Path(args.csv_out_dir) if args.csv_out_dir else out_dir
        write_selected_csvs(
            data_dir=data_dir,
            pattern=args.pattern,
            out_dir=csv_out_dir,
            selected_features=selected_features,
            label_col=args.label_col,
            keep_meta_cols=keep_meta_cols,
            suffix=args.csv_suffix,
        )
        print(f"Wrote reduced CSVs to: {csv_out_dir.resolve()} (suffix={args.csv_suffix})")


if __name__ == "__main__":
    main()
