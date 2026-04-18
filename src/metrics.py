import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


def cluster_accuracy(y_true, y_pred):

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()


    classes = np.unique(y_true)
    clusters = np.unique(y_pred)


    num_classes = len(classes)
    num_clusters = len(clusters)
    cost_matrix = np.zeros((num_classes, num_clusters), dtype=np.int64)

    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):

            cost_matrix[i, j] = np.sum((y_true == c) & (y_pred == k))

    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)


    correct_assignments = cost_matrix[row_ind, col_ind].sum()
    accuracy = correct_assignments / len(y_true)

    return accuracy


def evaluate_clustering(y_true, y_pred):

    y_true_flat = np.array(y_true).flatten()

    print("\n--- Evaluating Clustering Performance ---")


    acc = cluster_accuracy(y_true_flat, y_pred)
    print(f"Clustering Accuracy (ACC): {acc:.4f}")


    nmi = normalized_mutual_info_score(y_true_flat, y_pred)
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")

    ari = adjusted_rand_score(y_true_flat, y_pred)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    print("-----------------------------------------")

    return {"ACC": acc, "NMI": nmi, "ARI": ari}



if __name__ == "__main__":

    true_labels = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1, 1, 2, 2, 3]

    pred_clusters = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 2, 1, 0]

    results = evaluate_clustering(true_labels, pred_clusters)