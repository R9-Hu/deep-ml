import numpy as np

def repeat_rows(a: np.ndarray, d: int) -> np.ndarray:
    """Stack d copies of 1-D array a as rows."""
    # Your code here
    ones = (np.arange(d)*0.0+1.0)[:, None]
    return ones * a[None, :]
