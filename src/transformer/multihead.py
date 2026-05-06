import torch
import torch.nn as nn
from torch.nn import functional as F

from .head import SelfAttention_Head    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, 
        D_IN: int, 
        N_HEADS: int, 
        D_K: int,
    ):

        super().__init__()
        self.heads = nn.ModuleList([SelfAttention_Head(D_IN, D_K) for _ in range(N_HEADS)])

        """ the projection operator works as a weighted sum from the attention head outputs """
        self.proj = nn.Linear(D_K * N_HEADS , D_IN)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        """ concatenation of attention outputs as a single matrix """
        out = self.proj(out)
        return out