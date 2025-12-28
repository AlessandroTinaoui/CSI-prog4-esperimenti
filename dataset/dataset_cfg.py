DATASET = "new"  # "old" | "new" | "aug"

def get_train_path():
    if DATASET == "old":
        return "dataset/old_preprocessed_dataset/clients_dataset"
    elif DATASET == "new":
        return "dataset/new_preprocessed_dataset/clients_dataset"
    else:  # "aug"
        return "dataset/preprocessed_with_augmentation/clients_dataset"

def get_test_path():
    if DATASET == "old":
        return "dataset/old_preprocessed_dataset/x_test_clean.csv"
    elif DATASET == "new":
        return "dataset/new_preprocessed_dataset/x_test_clean.csv"
    else:  # "aug"
        return "dataset/preprocessed_with_augmentation/x_test_clean.csv"

