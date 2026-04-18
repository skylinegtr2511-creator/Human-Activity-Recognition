import numpy as np


def create_1d_hankel(signal, L):

    N = len(signal)
    K = N - L + 1

    if K <= 0:
        raise ValueError("Window length L cannot be greater than signal length N.")

    H = np.zeros((L, K))
    for i in range(L):
        H[i, :] = signal[i: i + K]

    return H


def sample_to_stacked_hankel(sample, L=64):

    num_channels = sample.shape[1]
    hankel_list = []

    for c in range(num_channels):

        signal_1d = sample[:, c]


        H_c = create_1d_hankel(signal_1d, L)
        hankel_list.append(H_c)


    stacked_H = np.vstack(hankel_list)
    return stacked_H


def embed_dataset(X, L=64):

    print(f"Embedding dataset of shape {X.shape} into Hankel matrices...")
    num_samples = X.shape[0]

    embeddings = []
    for i in range(num_samples):
        H_stacked = sample_to_stacked_hankel(X[i], L)
        embeddings.append(H_stacked)

    embeddings_array = np.array(embeddings)
    print(f"Finished. New embedded shape: {embeddings_array.shape}")
    return embeddings_array



if __name__ == "__main__":

    dummy_sample = np.random.randn(128, 9)


    L_window = 64
    stacked_matrix = sample_to_stacked_hankel(dummy_sample, L=L_window)

    print(f"Original sample shape: {dummy_sample.shape}")
    print(f"Stacked Hankel matrix shape: {stacked_matrix.shape}")
    print("\nBreakdown of dimensions:")
    print(f"Rows: 9 channels * {L_window} window length = {stacked_matrix.shape[0]}")
    print(f"Cols: 128 timesteps - {L_window} + 1 = {stacked_matrix.shape[1]}")