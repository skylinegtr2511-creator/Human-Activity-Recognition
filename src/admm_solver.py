import numpy as np


def flatten_embeddings(embeddings):

    num_samples = embeddings.shape[0]

    flattened = embeddings.reshape(num_samples, -1)


    return flattened.T


def soft_thresholding(x, kappa):

    return np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)


def admm_ssc(X, alpha=20.0, rho=1.0, max_iter=200, tol=1e-4):

    Features, Samples = X.shape
    print(f"Starting ADMM optimization on matrix of shape {X.shape}...")

    # Initialize variables
    C = np.zeros((Samples, Samples))
    Z = np.zeros((Samples, Samples))
    Lambda = np.zeros((Samples, Samples))


    XtX = X.T @ X
    I = np.eye(Samples)


    Z_update_inv = np.linalg.inv(alpha * XtX + rho * I)

    for iteration in range(max_iter):

        Z_new = Z_update_inv @ (alpha * XtX + rho * C - Lambda)

        np.fill_diagonal(Z_new, 0)

        C_new = soft_thresholding(Z_new + Lambda / rho, 1.0 / rho)
        np.fill_diagonal(C_new, 0)


        Lambda = Lambda + rho * (Z_new - C_new)

        primal_residual = np.linalg.norm(Z_new - C_new, ord='fro')

        Z = Z_new
        C = C_new

        if iteration % 10 == 0:
            print(f"  Iteration {iteration:3d} | Residual: {primal_residual:.6f}")

        if primal_residual < tol:
            print(f"Converged at iteration {iteration}! (Residual: {primal_residual:.6f})")
            break

    if iteration == max_iter - 1:
        print("Warning: ADMM reached max iterations without strict convergence.")

    return C


def build_affinity_matrix(C):

    C_abs = np.abs(C)
    W = C_abs + C_abs.T
    return W



if __name__ == "__main__":

    num_samples_test = 50
    dummy_embeddings = np.random.randn(num_samples_test, 576, 65)

    X_matrix = flatten_embeddings(dummy_embeddings)
    print(f"Flattened X shape: {X_matrix.shape} (Features, Samples)")


    coefficient_matrix = admm_ssc(X_matrix, alpha=10, rho=1.0, max_iter=50)


    affinity_matrix = build_affinity_matrix(coefficient_matrix)
    print(f"Affinity Matrix shape: {affinity_matrix.shape}")