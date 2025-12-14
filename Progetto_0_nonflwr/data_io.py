from __future__ import annotations
import pandas as pd

def load_client_merged_csv(path: str) -> pd.DataFrame:
    # I merged clean li hai salvati con to_csv default -> separatore ","
    return pd.read_csv(path)
