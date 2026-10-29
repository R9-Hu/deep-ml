import torch

def relu(t):
    """Element-wise ReLU: max(0, t).

    Args:
        t (torch.Tensor): input tensor

    Returns:
        torch.Tensor: activated tensor
    """
    # TODO: implement with pure torch ops
    return torch.maximum(torch.zeros(t.shape), t)

def leaky_relu(t, slope=0.01):
    """Element-wise Leaky ReLU with given negative slope.

    Args:
        t (torch.Tensor): input tensor
        slope (float): slope for negative values

    Returns:
        torch.Tensor: activated tensor
    """
    # TODO: implement with pure torch ops
    t_pos = torch.clamp(t, min=0, max=None)
    t_neg = torch.clamp(t, min=None, max=0)
    return t_pos + t_neg*slope
