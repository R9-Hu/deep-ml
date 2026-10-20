import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    # Your code here
    X = np.asarray(X)
    n_samples, n_features = X.shape
    X_cen = X - np.mean(X, axis=0, keepdims=True)
    Cov = X_cen.T @ X_cen/(n_samples-1)
    eigv = np.linalg.eigvalsh(Cov)
    ratios = np.sort(eigv)[::-1]/np.sum(eigv)
    return ratios.tolist()