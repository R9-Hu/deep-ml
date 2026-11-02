import numpy as np

def flatten(A: np.ndarray) -> np.ndarray:
    """Row-major flatten of 2-D array A without reshape/ravel."""
    # Your code here
    H, W = A.shape
    idx = np.arange(H * W)
    return A[idx // W, idx % W]
