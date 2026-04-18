import numpy as np
import os


def load_single_file(filepath):
    return np.loadtxt(filepath)


def load_group_signals(dataset_dir, group):
    signal_prefixes = [
        "body_acc_x_", "body_acc_y_", "body_acc_z_",
        "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
        "total_acc_x_", "total_acc_y_", "total_acc_z_"
    ]

    signals_dir = os.path.join(dataset_dir, group, "Inertial Signals")

    loaded_signals = []
    for prefix in signal_prefixes:
        filename = f"{prefix}{group}.txt"
        filepath = os.path.join(signals_dir, filename)


        signal_data = load_single_file(filepath)
        loaded_signals.append(signal_data)

    X = np.dstack(loaded_signals)
    return X


def load_group_labels(dataset_dir, group):

    filename = f"y_{group}.txt"
    filepath = os.path.join(dataset_dir, group, filename)

    y = load_single_file(filepath).reshape(-1, 1)
    return y


def load_dataset(dataset_dir):

    print(f"Loading data from: {dataset_dir}...")

    X_train = load_group_signals(dataset_dir, 'train')
    y_train = load_group_labels(dataset_dir, 'train')
    print(f"Train data loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    X_test = load_group_signals(dataset_dir, 'test')
    y_test = load_group_labels(dataset_dir, 'test')
    print(f"Test data loaded. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test



if __name__ == "__main__":

    dataset_path = os.path.join("..", "datasets", "UCI HAR Dataset")

    try:
        X_tr, y_tr, X_te, y_te = load_dataset(dataset_path)
        print("\nData loaded successfully! You are ready to start building Hankel matrices.")
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("Please check that your dataset_path is correct relative to where you run this script.")