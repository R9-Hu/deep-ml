import torch
import math
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    # TODO
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    N = X.shape[0]
    batch_size, batch_shape = 0, 0
    for batch_X, batch_y in loader:
        batch_size = batch_X.shape[0]
        batch_shape = batch_X.shape
        break
    return (math.ceil(N/batch_size), tuple(batch_shape))
