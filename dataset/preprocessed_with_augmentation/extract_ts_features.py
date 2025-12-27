from __future__ import annotations

import re
import json
import ast
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd


@dataclass
class TSFeatureConfig:
    # Se None: inferisce automaticamente le colonne TS
    ts_cols: Optional[List[str]] = None

    # Keyword per individuare colonne TS dal nome
    name_keywords: Tuple[str, ...] = ("time_series", "timeseries", "_ts", "ts_", "series")

    # Se colonna object ha stringhe mediamente molto lunghe -> candidata TS
    long_string_threshold: int = 200

    # Min lunghezza per calcolare dinamiche (slope/diff)
    min_len: int = 5

    # Se True elimina le colonne TS originali dopo aver estratto feature
    drop_original_ts_cols: bool = True

    # Prefisso per le nuove feature
    prefix: str = "ts"

    # Quantili da calcolare
    quantiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

    # ✅ Rimuove valori negativi prima di calcolare le feature (train + test)
    drop_negative_values: bool = True

    # ✅ aggiunge feature di qualità sulla TS raw (prima della pulizia)
    add_quality_features: bool = True

    # ✅ se la TS contiene troppi negativi (sulla raw), NON si usa (used=0 e feature NaN)
    max_neg_frac_raw: float = 0.50  # es. 0.50 = più del 50% negativi => ignora

    # ✅ dopo aver tolto negativi+NaN, se restano pochi punti => ignora
    min_valid_points: int = 5

_NUMS_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def sanitize_ts_array(y: np.ndarray, drop_negative: bool = True) -> np.ndarray:
    """
    Pulisce una time series prima della feature extraction.
    Se drop_negative=True, tutti i valori < 0 vengono trattati come invalidi e convertiti a NaN.
    """
    y = np.asarray(y, dtype=float)
    if drop_negative:
        y[y < 0] = np.nan
    return y


def _parse_ts_cell(x: Any) -> Optional[np.ndarray]:
    """
    Converte una cella (stringa/lista/json) in np.ndarray di float.
    Supporta:
      - liste python / array
      - stringhe che rappresentano liste: "[1,2,3]"
      - JSON array o dict con chiavi tipo values/ts/series/data
      - fallback: estrazione numeri dalla stringa
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return arr if arr.size else None

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None

    # JSON / literal python
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                arr = np.asarray(obj, dtype=float)
                return arr if arr.size else None
            if isinstance(obj, dict):
                for k in ("values", "ts", "series", "data"):
                    if k in obj and isinstance(obj[k], list):
                        arr = np.asarray(obj[k], dtype=float)
                        return arr if arr.size else None
        except Exception:
            pass

        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                arr = np.asarray(obj, dtype=float)
                return arr if arr.size else None
        except Exception:
            pass

    # fallback: estrai numeri dalla stringa
    s_norm = s
    if "." not in s_norm:
        s_norm = re.sub(r"(\d),(\d)", r"\1.\2", s_norm)

    s_norm = s_norm.replace(";", " ").replace("|", " ")
    nums = _NUMS_RE.findall(s_norm)
    if not nums:
        return None
    arr = np.asarray([float(v) for v in nums], dtype=float)
    return arr if arr.size else None



def _safe_slope(y: np.ndarray) -> float:
    """Slope di una regressione lineare y ~ t (t=0..n-1)."""
    n = y.size
    if n < 2:
        return np.nan
    t = np.arange(n, dtype=float)
    vt = np.var(t)
    if vt == 0:
        return np.nan
    return float(np.cov(t, y, bias=True)[0, 1] / vt)


def _extract_features_from_series(y: np.ndarray, cfg: TSFeatureConfig) -> Dict[str, float]:
    """
    - Calcola quality feats sulla serie raw
    - Se troppi negativi -> TS ignorata (used=0) e feature principali restano NaN
    - Altrimenti: negativi -> NaN (sempre), rimuove NaN, poi feature extraction
    """
    y = np.asarray(y, dtype=float)

    feats: Dict[str, float] = {}

    # --- Quality feats sempre disponibili (utile per debug/modello) ---
    if cfg.add_quality_features:
        feats["len_raw"] = float(y.size)
        feats["nan_frac_raw"] = float(np.mean(np.isnan(y))) if y.size else np.nan
        feats["neg_frac_raw"] = float(np.mean(y < 0)) if y.size else np.nan
        feats["neg_count_raw"] = float(np.sum(y < 0)) if y.size else np.nan

    # flag: 1 usata, 0 ignorata
    feats["used"] = 1.0

    # --- inizializza tutte le feature a NaN (schema stabile train/test) ---
    base_keys = ["len", "mean", "std", "min", "max", "range", "median", "mad", "energy"]
    for k in base_keys:
        feats[k] = np.nan
    for q in cfg.quantiles:
        feats[f"q{int(q * 100):02d}"] = np.nan
    dyn_keys = ["slope", "diff_mean", "diff_std", "diff_abs_mean"]
    for k in dyn_keys:
        feats[k] = np.nan

    # --- regola: se troppi negativi nella raw => ignora TS ---
    if y.size > 0:
        neg_frac = float(np.mean(y < 0))
        if neg_frac > cfg.max_neg_frac_raw:
            feats["used"] = 0.0
            return feats

    # --- pulizia: negativi -> NaN (sempre) e rimozione NaN ---
    y = sanitize_ts_array(y, drop_negative=cfg.drop_negative_values)
    y = y[~np.isnan(y)]

    # se dopo pulizia ci sono pochi punti, ignora
    if y.size < cfg.min_valid_points:
        feats["used"] = 0.0
        return feats

    # --- feature extraction ---
    feats["len"] = float(y.size)
    feats["mean"] = float(np.mean(y))
    feats["std"] = float(np.std(y))
    feats["min"] = float(np.min(y))
    feats["max"] = float(np.max(y))
    feats["range"] = float(np.max(y) - np.min(y))
    feats["median"] = float(np.median(y))
    feats["mad"] = float(np.median(np.abs(y - np.median(y))))
    feats["energy"] = float(np.mean(y * y))

    for q in cfg.quantiles:
        feats[f"q{int(q * 100):02d}"] = float(np.quantile(y, q))

    if y.size >= cfg.min_len:
        feats["slope"] = _safe_slope(y)
        dy = np.diff(y)
        feats["diff_mean"] = float(np.mean(dy)) if dy.size else np.nan
        feats["diff_std"] = float(np.std(dy)) if dy.size else np.nan
        feats["diff_abs_mean"] = float(np.mean(np.abs(dy))) if dy.size else np.nan

    return feats



def infer_ts_columns(df: pd.DataFrame, cfg: TSFeatureConfig) -> List[str]:
    """Individua colonne candidate TS per nome e/o stringhe lunghe."""
    cols = list(df.columns)

    by_name = [c for c in cols if any(k in c.lower() for k in cfg.name_keywords)]

    by_len: List[str] = []
    for c in cols:
        if df[c].dtype != "object":
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        if s.astype(str).map(len).mean() > cfg.long_string_threshold:
            by_len.append(c)

    return sorted(set(by_name + by_len))


def extract_ts_features(df: pd.DataFrame, cfg: TSFeatureConfig) -> pd.DataFrame:
    """
    Estrae feature tabellari dalle colonne TS e opzionalmente rimuove le colonne TS originali.
    - Per ogni colonna TS crea nuove colonne: f"{prefix}__{col}__<feat>"
    - ✅ Prima di calcolare le feature, elimina i valori negativi (se cfg.drop_negative_values=True)
    """
    out = df.copy()

    ts_cols = cfg.ts_cols if cfg.ts_cols is not None else infer_ts_columns(out, cfg)
    if not ts_cols:
        return out

    feat_frames: List[pd.DataFrame] = []
    for c in ts_cols:
        rows_feats: List[Dict[str, float]] = []
        for x in out[c].tolist():
            arr = _parse_ts_cell(x)
            if arr is None:
                arr = np.array([], dtype=float)
            rows_feats.append(_extract_features_from_series(arr, cfg))

        fdf = pd.DataFrame(rows_feats).add_prefix(f"{cfg.prefix}__{c}__")
        feat_frames.append(fdf)

    feats_all = pd.concat(feat_frames, axis=1)
    out = pd.concat([out.reset_index(drop=True), feats_all.reset_index(drop=True)], axis=1)

    if cfg.drop_original_ts_cols:
        out = out.drop(columns=ts_cols, errors="ignore")

    return out
