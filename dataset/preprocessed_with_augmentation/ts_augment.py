# ts_augment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ri-uso la tua logica di parsing senza dipendenze circolari.
# Se preferisci, puoi importare _parse_ts_cell dal tuo extract_ts_features.py.
import re, json, ast
_NUMS_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_ts_cell(x: Any) -> Optional[np.ndarray]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return arr if arr.size else None

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None

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

    s_norm = s
    if "." not in s_norm:
        s_norm = re.sub(r"(\d),(\d)", r"\1.\2", s_norm)
    s_norm = s_norm.replace(";", " ").replace("|", " ")
    nums = _NUMS_RE.findall(s_norm)
    if not nums:
        return None
    arr = np.asarray([float(v) for v in nums], dtype=float)
    return arr if arr.size else None


def format_ts_cell(arr: np.ndarray) -> str:
    # Salva come JSON array
    arr = np.asarray(arr, dtype=float)
    return json.dumps([float(x) for x in arr.tolist()])


def _time_warp_simple(arr: np.ndarray, rng: np.random.Generator, warp_strength: float) -> np.ndarray:
    """
    Time-warp semplice: ricampiona su una griglia "stirata" leggermente.
    Non richiede scipy.
    """
    n = arr.size
    if n < 4:
        return arr.copy()

    # costruiamo una griglia monotona con piccole variazioni
    t = np.linspace(0.0, 1.0, n)
    noise = rng.normal(0.0, warp_strength, size=n)
    noise = np.cumsum(noise)
    noise -= noise.min()
    if noise.max() > 0:
        noise /= noise.max()
    t_warp = 0.85 * t + 0.15 * noise  # mantiene monotonia "quasi"
    t_warp = np.clip(t_warp, 0.0, 1.0)
    t_warp = np.maximum.accumulate(t_warp)

    # interp
    x_new = np.interp(t, t_warp, arr)
    return x_new


@dataclass
class TSAugmentConfig:
    enabled: bool = True
    # probabilità di applicare augmentation a una riga (giorno)
    p_row: float = 0.35
    # probabilità di applicare augmentation a ciascuna colonna TS
    p_col: float = 0.60

    # jitter: rumore additivo proporzionale alla std
    jitter_std_frac: float = 0.02

    # scaling moltiplicativo (1 +/- scaling_frac)
    scaling_frac: float = 0.03

    # dropout punti (imposta alcuni punti a NaN)
    dropout_frac: float = 0.02

    # time warp semplice
    time_warp_strength: float = 0.015

    # seed per riproducibilità
    seed: int = 42

    # se True, non altera valori negativi: lascia gestire a extract_ts_features
    keep_negatives: bool = True


def augment_ts_dataframe(
    df: pd.DataFrame,
    ts_cols: List[str],
    cfg: TSAugmentConfig,
) -> pd.DataFrame:
    """
    Augmenta le TS nel DF (celle che sono liste/stringhe) e restituisce DF modificato.
    Mantiene schema colonne invariato.
    """
    if not cfg.enabled or not ts_cols:
        return df

    out = df.copy()
    rng = np.random.default_rng(cfg.seed)

    for idx in range(len(out)):
        if rng.random() > cfg.p_row:
            continue

        for c in ts_cols:
            if c not in out.columns:
                continue
            if rng.random() > cfg.p_col:
                continue

            arr = parse_ts_cell(out.at[idx, c])
            if arr is None or arr.size < 5:
                continue

            arr_aug = arr.astype(float).copy()

            # jitter
            std = float(np.nanstd(arr_aug)) if np.isfinite(arr_aug).any() else 0.0
            if std > 0:
                arr_aug = arr_aug + rng.normal(0.0, cfg.jitter_std_frac * std, size=arr_aug.size)

            # scaling
            scale = 1.0 + rng.uniform(-cfg.scaling_frac, cfg.scaling_frac)
            arr_aug = arr_aug * scale

            # time warp
            arr_aug = _time_warp_simple(arr_aug, rng, cfg.time_warp_strength)

            # dropout (NaN)
            if cfg.dropout_frac > 0:
                m = int(np.floor(cfg.dropout_frac * arr_aug.size))
                if m > 0:
                    drop_idx = rng.choice(arr_aug.size, size=m, replace=False)
                    arr_aug[drop_idx] = np.nan

            if not cfg.keep_negatives:
                arr_aug[arr_aug < 0] = np.nan

            out.at[idx, c] = format_ts_cell(arr_aug)

    return out
