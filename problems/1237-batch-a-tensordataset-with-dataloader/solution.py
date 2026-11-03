import torch
import math
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    num_batches = 0
    first_batch_shape = None

    for batch_X, batch_y in loader:
        if num_batches == 0:
            first_batch_shape = tuple(batch_X.shape)
        num_batches += 1

    return num_batches, first_batch_shape
