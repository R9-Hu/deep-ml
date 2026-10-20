import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    X_cen = X - X.mean(axis=0, keepdims=True)
    X_cov = X_cen.T @ X_cen / n_samples
    eigvals = np.linalg.eigvalsh(X_cov)
    idxs = np.argsort(eigvals)[::-1]
    
    return np.sum(eigvals[idxs[n_components:]])/n_features


