import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

def load_data(partition_id: int, root_dir: str = "clients_data"):
    """Load data for a specific client partition."""
    file_path = os.path.join(root_dir, f"clients_data/group{partition_id}_merged_clean.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    # Load data
    data = pd.read_csv(file_path, sep=',')
    
    # Drop unnecessary columns
    cols_to_drop = ["day", "client_id", "user_id", "source_file"]
    cols_to_drop = [c for c in cols_to_drop if c in data.columns]
    
    data = data.dropna()
    
    X = data.drop(columns=cols_to_drop + ["label"])
    y = data["label"]
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def get_model(n_estimators: int = 50, max_depth: int = 10):
    """Create Random Forest regressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=1
    )
