import torch
import torch.nn as nn
from torch.nn import functional as F

from .multihead import MultiHeadAttention
from .mlp import MLP

device = "cuda" if torch.cuda.is_available() else "cpu"

class SingleLayerTransformer(nn.Module):
    """
    TW(X) = MW( A(X) )
 
    The model prediction for a prompt [C, x] is the D-th component
    of the last token output (Eq. 11 in the paper):
        y_hat = TW(E_tau)[d+1, N+1]
    """
    def __init__(self, D_IN, D_MLP, N_HEADS, D_K):
        super().__init__()
        self.attn = MultiHeadAttention(D_IN, N_HEADS, D_K).to(device)
        self.mlp  = MLP(D_MLP, D_IN).to(device)
 
    def forward(self, X, W_override=None):
        # X: (B, T, D_IN)  →  (B, T, D_IN)
        return self.mlp(self.attn(X), W_override=W_override)