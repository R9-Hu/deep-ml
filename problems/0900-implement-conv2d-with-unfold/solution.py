import torch
import torch.nn.functional as F

def conv2d_via_unfold(x, W, b, stride=1, padding=0):
    # TODO: implement conv2d using F.unfold + matmul
    N,_,height, width = x.shape
    x_unfold = F.unfold(x,(W.shape[2],W.shape[3]),padding=padding,stride=stride)
    L = x_unfold.shape[-1]
    x_unfold = x_unfold.transpose(1,2)
    C_out = W.shape[0]
    W_unfold = W.reshape(C_out,-1)
    x_out = x_unfold @ W_unfold.T + b
    Hout = (height+2*padding-W.shape[2])//stride+1
    Wout = (width+2*padding-W.shape[3])//stride+1
    x_out = x_out = x_out.transpose(1, 2).reshape(N, C_out, Hout, Wout)

    return x_out

