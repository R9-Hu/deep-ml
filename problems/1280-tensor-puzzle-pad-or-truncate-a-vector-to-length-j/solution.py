import numpy as np

def pad_to(a: np.ndarray, j: int) -> np.ndarray:
    """Pad a with zeros (or truncate) to length j."""
    # Your code here
    padding = np.zeros(j)
    padding[:np.minimum(a.shape[0],j)] = a[:np.minimum(a.shape[0],j)]
    return padding
