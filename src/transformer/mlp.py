import torch
import torch.nn as nn
from torch.nn import functional as F

class MLP(nn.Module):
    """
    MW(x) = Wp · relu(W·x + b) + bp
 
    W is the first-layer weight — the one implicitly updated by the context
    via the rank-1 formula in Theorem 2.2.
    """
    def __init__(self, D_MLP, D_IN):
        super().__init__()
        self.W  = nn.Parameter(torch.randn(D_MLP, D_IN) * 0.02)
        self.b  = nn.Parameter(torch.zeros(D_MLP))
        self.Wp = nn.Parameter(torch.randn(D_IN, D_MLP) * 0.02)
        self.bp = nn.Parameter(torch.zeros(D_IN))
 
    def forward(self, x, W_override=None):
        # x: (B, T, D_IN)
        W = W_override if W_override is not None else self.W
        return F.relu(x @ W.T + self.b) @ self.Wp.T + self.bp
 