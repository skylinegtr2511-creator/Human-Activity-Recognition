import numpy as np
from sklearn.cluster import SpectralClustering
import warnings
warnings.filterwarnings("ignore")


def perform_clustering(affinity_matrix, n_clusters=6):

    print(f"Running Spectral Clustering to find {n_clusters} clusters...")

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        n_init=10
    )

    predicted_labels = sc.fit_predict(affinity_matrix)
    print("Clustering complete!")

    return predicted_labels



if __name__ == "__main__":

    num_samples_test = 50


    dummy_W = np.random.rand(num_samples_test, num_samples_test)


    dummy_W = (dummy_W + dummy_W.T) / 2
    np.fill_diagonal(dummy_W, 0)


    labels = perform_clustering(dummy_W, n_clusters=6)

    print(f"\nAssigned labels shape: {labels.shape}")
    print(f"First 15 assigned labels: {labels[:15]}")