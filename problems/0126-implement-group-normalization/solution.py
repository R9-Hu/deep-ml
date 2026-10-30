import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    B, C, H, W = X.shape
    groupSize = C//num_groups
    X_re = X.reshape(B, num_groups, groupSize, H, W)
    mean_val = X_re.mean(axis=(2,3,4), keepdims=True)
    var_val = X_re.var(axis=(2,3,4), keepdims=True)
    X_re_norm = (X_re - mean_val)/np.sqrt(var_val+epsilon)
    X_norm = X_re_norm.reshape(B,C,H,W)
    return X_norm*gamma+beta