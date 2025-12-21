DATASET = "new"


def get_train_path():
    if DATASET == "old":
        return "dataset/old_preprocessed_dataset/clients_dataset"
    else: #dataset == "new"
        return "dataset/new_preprocessed_dataset/clients_dataset"

def get_test_path():
    if DATASET == "old":
        return "dataset/old_preprocessed_dataset/x_test_clean.csv"
    else:
        return "dataset/new_preprocessed_dataset/x_test_clean.csv"