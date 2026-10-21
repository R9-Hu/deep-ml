import numpy as np

def randomized_pca(X: np.ndarray, k: int, n_oversamples: int = 5, n_iter: int = 2, seed: int = 0) -> np.ndarray:
    """Approximate top-k PCA projection of X using a randomized algorithm.

    Returns a numpy array of shape (n, k): the centered data projected onto
    the top-k principal directions, with a deterministic sign convention.
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)
    sample_size = min(k+n_oversamples, min(n, d))
    Y = X_c[:,rng.randint(0,d,sample_size)]
    for _ in range(n_iter):
        # (n, sample_size)
        Y = X_c @ (X_c.T @ Y)
    Q, R = np.linalg.qr(Y)
    B = Q.T @ X_c
    _, _, V = np.linalg.svd(B, full_matrices=False)
    
    directions = V[:k, :].copy()
    for direction in directions:
        idx = np.argmax(np.abs(direction))
        if direction[idx] < 0:
            direction *= -1

    return X_c @ directions.T


