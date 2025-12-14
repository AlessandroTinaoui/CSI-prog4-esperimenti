from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    label_col: str = "label"
    day_col: Optional[str] = "day"
    drop_cols: tuple[str, ...] = ("client_id", "user_id", "source_file")


@dataclass
class FeatureSelectionConfig:
    k_top: int = 30
    scoring: str = "neg_root_mean_squared_error"  # oppure "neg_mean_absolute_error"
    n_repeats: int = 5
    val_size: float = 0.2
    random_state: int = 42


@dataclass
class RFLocalConfig:
    n_estimators: int = 80
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class FederatedConfig:
    rounds: int = 5
    max_global_trees: int = 400
