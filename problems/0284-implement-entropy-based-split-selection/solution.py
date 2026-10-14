import numpy as np

def entropy_split_selection(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Find the best feature and threshold for splitting based on information gain.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
    
    Returns:
        Tuple of (best_feature_index, best_threshold, best_info_gain)
    """
    def entropy(y):
        _, cnts = np.unique(y, return_counts=True)
        prob = cnts/np.sum(cnts)
        return -np.sum(prob*np.log2(prob))
    
    best_gain = 0
    res = [0, 1.5, 0]
    n_samples, n_features = X.shape
    for i in range(n_features):
        order = np.argsort(X[:, i], kind="stable")

        X_copy = X[order]
        y_copy = y[order]
        entropy_all = entropy(y_copy)
        for j in range(1, n_samples):
            if X_copy[j,i] == X_copy[j-1, i]:
                continue

            gain = entropy_all - (j*entropy(y_copy[:j])+(n_samples-j)*entropy(y_copy[j:]))/n_samples
            if gain > best_gain:
                best_gain = gain
                threshold = (X_copy[j-1,i]+X_copy[j,i])/2
                res = [i, threshold, best_gain]
    return res
