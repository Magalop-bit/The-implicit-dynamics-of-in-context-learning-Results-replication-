import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class SelfAttention_Head(nn.Module):
    """ one head of self-attention """

    def __init__(self,
        D_IN: int, 
        D_K: int,
    ):
        
        """
        D_IN: embedding dimention
        D_K: context length for attention

        """
        super().__init__()
        self.key = nn.Linear(D_IN, D_K, bias=False).to(device)
        self.query = nn.Linear(D_IN, D_K, bias=False).to(device)
        self.value = nn.Linear(D_IN, D_K, bias=False).to(device)
        

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape

        k = self.key(x)   # (B, T, D_K)
        q = self.query(x) # (B, T, D_K)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, D_K) @ (B, D_K, T) -> (B, T, T)

        # Create causal mask on-the-fly (not registered as buffer to avoid state_dict issues)
        tril = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(tril == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,D_K)
        out = wei @ v # (B, T, T) @ (B, T, D_K) -> (B, T, D_K)
        return out