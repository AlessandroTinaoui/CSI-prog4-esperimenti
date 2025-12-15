import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_data(partition_id: int, num_partitions: int):
    """Load and partition data for federated learning."""
    from sklearn.datasets import load_digits
    
    # Carica il dataset
    data = load_digits()
    X, y = data.data, data.target
    
    # Partiziona i dati tra i client
    X_parts = np.array_split(X, num_partitions)
    y_parts = np.array_split(y, num_partitions)
    
    X_local = X_parts[partition_id]
    y_local = y_parts[partition_id]
    
    # Split train/test per questo client
    X_train, X_test, y_train, y_test = train_test_split(
        X_local, y_local, test_size=0.25, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def get_model(n_estimators: int = 50, max_depth: int = None):
    """Create Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
