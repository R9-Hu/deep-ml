import numpy as np

def sequence_mask(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Zero out positions j >= lengths[i] in each row i."""
    # Your code here
    pos = np.arange(values.shape[1])[None, :]
    mask = pos < lengths[:, None]
    return values * mask

