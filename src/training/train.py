import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.transformer import SingleLayerTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ── Hyperparameters ──────────────────────────────────────────
D       = 2                    # feature dimension
N       = 100                  # in-context examples per prompt
D_MLP   = 128                  # MLP hidden dim
D_MODEL = 32                   # attention output dim
N_HEADS = 8
D_K     = D_MODEL // N_HEADS   # per-head dim = 4
D_IN    = D + 1  

def sample_batch(batch_size, device):
    """
    Returns:
        prompts : (B, N+1, D+1)  — context tokens + query token
        targets : (B,)           — true label for the query
    """
    omega   = torch.randn(batch_size, D, device=device)
    X       = torch.randn(batch_size, N, D, device=device)
    xquery  = torch.randn(batch_size, D, device=device)
 
    y_ctx   = (X * omega.unsqueeze(1)).sum(-1)              # (B, N)
    y_query = (xquery * omega).sum(-1)                      # (B,)
 
    ctx   = torch.cat([X, y_ctx.unsqueeze(-1)], dim=-1)     # (B, N,   D+1)
    query = torch.cat([xquery, torch.zeros(batch_size, 1, device=device)], dim=-1).unsqueeze(1)  # (B, 1, D+1)
 
    prompts = torch.cat([ctx, query], dim=1)                # (B, N+1, D+1)
    return prompts, y_query

def train(n_steps=2000, batch_size=64, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SingleLayerTransformer(D_IN, D_MLP, N_HEADS, D_K).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
 
    for step in range(1, n_steps + 1):
        prompts, targets = sample_batch(batch_size, device)
 
        # Prediction: D-th component of the last token output (Eq. 11)
        pred = model(prompts)[:, -1, D]
        loss = 0.5 * F.mse_loss(pred, targets)
 
        opt.zero_grad()
        loss.backward()
        opt.step()
 
        if step % 20 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")
 
    return model

def test(model, n_tasks=100):
    """
    Evaluate the trained model on held-out tasks.
    Reports MSE on the query prediction.
    """
    model.eval()
    with torch.no_grad():
        prompts, targets = sample_batch(n_tasks, device)
        pred = model(prompts)[:, -1, D]
        loss = 0.5 * F.mse_loss(pred, targets)
        print(f"\nTest | tasks {n_tasks} | loss {loss.item():.4f}")
    return loss.item()

if __name__ == "__main__":
    model = train()
    test(model)


