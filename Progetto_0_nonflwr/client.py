from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from Progetto_0_nonflwr.config import DataConfig, FeatureSelectionConfig, RFLocalConfig
from Progetto_0_nonflwr.feature_selection import local_permutation_importance
from Progetto_0_nonflwr.rf_federated import train_local_rf


@dataclass
class FSResult:
    client_id: str
    n_samples: int
    importances: Dict[str, float]


@dataclass
class TrainResult:
    client_id: str
    n_samples: int
    estimators: list


class FederatedClient:
    def __init__(self, client_id: str, df: pd.DataFrame, dc: DataConfig):
        self.client_id = client_id
        self.df = df
        self.dc = dc

    def compute_importances(self, rf_cfg: RFLocalConfig, fs_cfg: FeatureSelectionConfig, features: List[str]) -> FSResult:
        imps = local_permutation_importance(self.df, self.dc, rf_cfg, fs_cfg, features)
        return FSResult(self.client_id, int(len(self.df)), imps)

    def train_local(self, rf_cfg: RFLocalConfig, selected_features: List[str]) -> TrainResult:
        model = train_local_rf(self.df, self.dc, rf_cfg, selected_features)
        return TrainResult(self.client_id, int(len(self.df)), list(model.estimators_))
