import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import os


def plot_affinity_matrix(W, save_path=None):

    plt.figure(figsize=(8, 6))


    plt.imshow(np.log(W + 1e-10), cmap='viridis', interpolation='nearest')

    plt.colorbar(label='Log Affinity')
    plt.title('ADMM Affinity Matrix (Log Scale)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')

    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Affinity matrix plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def align_predicted_labels(y_true, y_pred):

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    classes = np.unique(y_true)
    clusters = np.unique(y_pred)

    cost_matrix = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            cost_matrix[i, j] = np.sum((y_true == c) & (y_pred == k))

    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)


    mapping = {clusters[c]: classes[r] for r, c in zip(row_ind, col_ind)}


    aligned_preds = np.array([mapping[val] for val in y_pred])
    return aligned_preds



if __name__ == "__main__":

    block1 = np.ones((20, 20))
    block2 = np.ones((20, 20)) * 0.5


    from scipy.linalg import block_diag

    dummy_W = block_diag(block1, block2)

    dummy_W += np.random.rand(40, 40) * 0.1

    print("Testing visualization function...")
    plot_affinity_matrix(dummy_W)