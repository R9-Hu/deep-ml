import numpy as np

def repeat_rows(a: np.ndarray, d: int) -> np.ndarray:
    """Stack d copies of 1-D array a as rows."""
    # Your code here
    A_rep = np.repeat(a[np.newaxis, :], d, axis=0)
    return A_rep
