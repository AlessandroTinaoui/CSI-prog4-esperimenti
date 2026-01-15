# CSI-prog4 Esperimenti

Federated learning project implementing multiple machine learning models (MLP, TabNet, XGBoost, Random Forest, ExtraTrees) for distributed training.

## Main Directory Structure

- **`/mlp/`**, **`/TabNet/`**, **`/xgboostmodel/`**, **`/randomforest/`**, **`/extratreemodel/`**: Each model has its own federated learning implementation with client and server components.
  - `client/`: Federated client with training logic
  - `server/`: Federated server with aggregation strategy
- **`/dataset/`**: Dataset storage and preprocessing pipelines.
- **`/tuning/`**: Hyperparameter optimization framework.
- **`run_train.py`**: Main training orchestrator script.
- **`visualize_study.py`**: Visualization for hyperparameter tuning results.

## Preprocessing

The preprocessing pipeline ensures all models can use the same set of features:
- **Raw data** in `/dataset/raw_dataset/` organized by groups (group0-group8).
- **Multiple preprocessing versions** available:
  - `/dataset/new_preprocessed_dataset/`: Latest version with cleaned features
  - `/dataset/preprocessed_with_augmentation/`: Data with augmentation techniques
  - `/dataset/preprocessing_2/` & `/dataset/preprocessing_3/`: Alternative preprocessing approaches
- Each preprocessing script (e.g., `extract_ts_features.py`) extracts and standardizes time series features so they can be consumed by all models consistently.
- Processed data is grouped by client in `clients_dataset/` (group0-group8) with corresponding test sets.

## run_train.py

Main training orchestrator that:
- Supports all available models via command-line arguments
- Manages federated server and client processes
- Configures training rounds, number of clients, and model-specific hyperparameters
- Aggregates model updates across federated clients using Flower framework

## Tuning

Hyperparameter optimization framework using Optuna:
- **`search_space.py`**: Defines hyperparameter search spaces for each model.
- **`study_driver.py`**: Orchestrates the hyperparameter optimization process.
- **`run_holdout_benchmark.py`**: Validates model performance with holdout sets.
- **`trial_io.py`**: Manages study persistence and trial data I/O.
- **`visualize_study.py`**: Visualizes optimization history and parameter importance.
