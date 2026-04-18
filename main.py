import os
import time
import numpy as np

# Import all our custom modules from the 'src' folder
from src.data_loader import load_dataset
from src.hankel_embedding import embed_dataset
from src.admm_solver import flatten_embeddings, admm_ssc, build_affinity_matrix
from src.clustering import perform_clustering
from src.metrics import evaluate_clustering
from src.utils import plot_affinity_matrix


def main():
    print("==================================================")
    print("Human Activity Recognition via ADMM & Hankel Matrices")
    print("==================================================\n")

    # 1. Setup Paths
    # Points to the UCI HAR Dataset folder
    dataset_path = os.path.join("datasets", "UCI HAR Dataset")
    results_dir = os.path.join("results", "figures")
    os.makedirs(results_dir, exist_ok=True)

    # 2. Load the Raw Data
    print("[STEP 1/6] Loading Dataset...")
    X_train, y_train, X_test, y_test = load_dataset(dataset_path)

    # --- DATA SUBSETTING (CORRECTED BALANCED BATCH) ---
    print("\n[INFO] Creating a balanced subset (20 samples per activity)...")

    X_list = []
    y_list = []

    # Loop through the 6 activity labels (1 to 6)
    for class_id in range(1, 7):
        # Find all indices where the label matches the current class
        class_indices = np.where(y_train.flatten() == class_id)[0]
        # Grab exactly the first 20 samples of this class
        selected_indices = class_indices[:20]

        X_list.append(X_train[selected_indices])
        y_list.append(y_train[selected_indices])

    # Stack them back together.
    # Because we appended 1s, then 2s, etc., the data is perfectly sorted!
    X_subset = np.concatenate(X_list, axis=0)
    y_subset = np.concatenate(y_list, axis=0)

    # 3. Hankel Matrix Embedding
    print("\n[STEP 2/6] Generating Hankel Embeddings...")
    start_time = time.time()
    embeddings = embed_dataset(X_subset, L=64)
    print(f"Embedding completed in {time.time() - start_time:.2f} seconds.")

    # 4. ADMM Optimization (Sparse Subspace Clustering)
    print("\n[STEP 3/6] Running ADMM Optimization...")
    X_flattened = flatten_embeddings(embeddings)

    # --- NEW FIX: Normalize the columns to unit L2 norm ---
    # This prevents the soft-thresholding from wiping out your data
    col_norms = np.linalg.norm(X_flattened, axis=0)
    X_flattened = X_flattened / (col_norms + 1e-10)

    start_time = time.time()

    # We also tweak the parameters. Lowering alpha allows for more flexibility.
    # Increasing rho lowers the thresholding penalty.
    coefficient_matrix = admm_ssc(X_flattened, alpha=50.0, rho=2.0, max_iter=100)

    print(f"ADMM completed in {time.time() - start_time:.2f} seconds.")

    print("\n[STEP 4/6] Building Affinity Graph...")
    affinity_matrix = build_affinity_matrix(coefficient_matrix)

    # 5. Spectral Clustering
    print("\n[STEP 5/6] Performing Spectral Clustering...")
    predicted_clusters = perform_clustering(affinity_matrix, n_clusters=6)

    # 6. Evaluation and Visualization
    print("\n[STEP 6/6] Evaluating and Plotting Results...")
    metrics = evaluate_clustering(y_subset, predicted_clusters)

    plot_path = os.path.join(results_dir, "admm_affinity_matrix.png")
    plot_affinity_matrix(affinity_matrix, save_path=plot_path)

    print("\n==================================================")
    print("Pipeline Execution Complete!")
    print(f"Results saved to: {results_dir}")
    print("==================================================")


if __name__ == "__main__":
    main()