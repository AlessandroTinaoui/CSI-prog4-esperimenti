from mlp.server.config import HOLDOUT_CID
DATASET = "three"  # "old" | "new" | "aug" | "two"|

HOLDOUT = HOLDOUT_CID
def get_train_path():
    if DATASET == "old":
        return "dataset/old_preprocessed_dataset/clients_dataset"
    elif DATASET == "new":
        return "dataset/new_preprocessed_dataset/clients_dataset"
    elif DATASET == "two":
        return "dataset/preprocessing_2/clients_dataset"
    elif DATASET == "three":
        return "dataset/preprocessing_3/clients_dataset"
    else:  # "aug"
        return "dataset/preprocessed_with_augmentation/clients_dataset"

def get_test_path():
    if DATASET == "old":
        return "dataset/old_preprocessed_dataset/x_test_clean.csv"
    elif DATASET == "new":
        return "dataset/new_preprocessed_dataset/x_test_clean.csv"
    elif DATASET == "two":
        return "dataset/preprocessing_2/x_test_clean.csv"
    elif DATASET == "three":
        return "dataset/preprocessing_3/clients_dataset/x_test_clean.csv"
    else:  # "aug"
        return "dataset/preprocessed_with_augmentation/x_test_clean.csv"

def get_script_path():
    if DATASET == "old":
        return "dataset/old_preprocessed_dataset/old_preprocessing.py"
    elif DATASET == "new":
        return "dataset/new_preprocessed_dataset/client_dataset_setup.py"
    elif DATASET == "two":
        return "dataset/preprocessing_2/preprocess2_global.py"
    elif DATASET == "three":
        return "dataset/preprocessing_3/preprocessing_3.py"
    else:
        return "dataset/preprocessed_with_augmentation/preprocess_global.py"