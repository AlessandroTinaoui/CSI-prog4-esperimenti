import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

def load_data(partition_id: int, root_dir: str = "clients_data"):
    file_path = os.path.join(root_dir, f"clients_data/group{partition_id}_merged_clean.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = pd.read_csv(file_path, sep=",").dropna()

    cols_to_drop = ["day", "client_id", "user_id", "source_file"]
    cols_to_drop = [c for c in cols_to_drop if c in data.columns]

    X = data.drop(columns=cols_to_drop + ["label"])
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    return X_train, X_test, y_train, y_test


def get_model(
    n_estimators: int = 300,
    max_depth=None,
    random_state: int = 42,
    n_jobs: int = 1,
    max_features=1.0,
    bootstrap: bool = False,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
):
    """Create ExtraTrees regressor."""
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        max_features=max_features,
        bootstrap=bootstrap,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
