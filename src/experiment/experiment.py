import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from transformer.transformer import SingleLayerTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ── Hyperparameters ──────────────────────────────────────────
D       = 2                    # feature dimension
N       = 100 #100                  # in-context examples per prompt
D_MLP   = 128                  # MLP hidden dim
D_MODEL = 32                   # attention output dim
N_HEADS = 8
D_K     = D_MODEL // N_HEADS   # per-head dim = 4
D_IN    = D + 1  

def sample_batch(batch_size: int, context_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample batch of prompts with linear function tasks.

    Args:
        batch_size: Number of tasks
        context_len: Number of in-context examples
        device: cuda or cpu

    Returns:
        prompts: (B, context_len+1, D+1) — context + query
        targets: (B,) — true labels
    """
    omega = torch.randn(batch_size, D, device=device)
    X = torch.randn(batch_size, context_len, D, device=device)
    x_query = torch.randn(batch_size, D, device=device)

    y_ctx = (X * omega.unsqueeze(1)).sum(-1)  # (B, context_len)
    y_query = (x_query * omega).sum(-1)         # (B,)

    ctx = torch.cat([X, y_ctx.unsqueeze(-1)], dim=-1)  # (B, context_len, D+1)
    query = torch.cat(
        [x_query, torch.zeros(batch_size, 1, device=device)],
        dim=-1
    ).unsqueeze(1)  # (B, 1, D+1)

    prompts = torch.cat([ctx, query], dim=1)  # (B, context_len+1, D+1)
    return prompts, y_query

def compute_implicit_weight_update(
    model: SingleLayerTransformer,
    context_tokens: torch.Tensor,  # (1, N, D_IN)
    x_query: torch.Tensor,         # (1, 1, D_IN)
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        full_prompt = torch.cat([context_tokens, x_query], dim=1)  # (1, N+1, D_IN)

        a_cx = model.attn(full_prompt)[:, -1, :]       # (1, D_IN)
        a_x  = model.attn(x_query)[:, -1, :]           # (1, D_IN)

        delta_a = a_cx - a_x                            # (1, D_IN)
        W       = model.mlp.W                           # (D_MLP, D_IN)

        w_delta_a = W @ delta_a.T                       # (D_MLP, 1)
        norm_sq   = (a_x ** 2).sum()                    # scalar
        delta_w   = w_delta_a @ a_x / norm_sq           # (D_MLP, D_IN)  rank-1

    return delta_w

def experiment_verify_theorem_22(
    model: SingleLayerTransformer,
    n_eval: int = 100,
) -> dict:
    model.eval()
    results = {
        'context_lens': [],
        'loss_with_context': [],
        'loss_with_delta_w': [],
        'max_diff': [],
    }

    for ctx_len in range(1, N + 1, max(1, N // 20)):
        preds_context = []
        preds_delta_w = []

        with torch.no_grad():
            prompts, targets = sample_batch(n_eval, ctx_len, device)

            # Method 1: standard forward with full context
            output_full = model(prompts)
            pred_with_context = output_full[:, -1, D]

            # Method 2: per-task ΔW
            for i in range(n_eval):
                context = prompts[i:i+1, :ctx_len, :]
                x_query = prompts[i:i+1, ctx_len:,  :]
                delta_w = compute_implicit_weight_update(model, context, x_query)
                w_mod   = model.mlp.W + delta_w

                a_q      = model.attn(x_query)
                pred_mod = model(x_query, W_override=w_mod)
                preds_delta_w.append(pred_mod[0, -1, D])

            preds_delta_w = torch.stack(preds_delta_w)

            loss_context = 0.5 * F.mse_loss(pred_with_context, targets).item()
            loss_delta_w = 0.5 * F.mse_loss(preds_delta_w,     targets).item()
            max_diff     = (pred_with_context - preds_delta_w).abs().max().item()

            results['context_lens'].append(ctx_len)
            results['loss_with_context'].append(loss_context)
            results['loss_with_delta_w'].append(loss_delta_w)
            results['max_diff'].append(max_diff)

        #print(f"ctx_len {ctx_len:4d} | loss_ctx {loss_context:.4f} | loss_ΔW {loss_delta_w:.4f} | max_diff {max_diff:.2e}")

    return results

# ─────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────

def plot_theorem_verification(results: dict, save_path: str = ""):
    """Plot results from Theorem 2.2 verification."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Loss comparison
    axes[0].plot(results['context_lens'], results['loss_with_context'],
                 label='With context', marker='o')
    axes[0].plot(results['context_lens'], results['loss_with_delta_w'],
                 label='With ΔW', marker='s')
    axes[0].set_xlabel('Context length')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Theorem 2.2: Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Max prediction difference
    axes[1].plot(results['context_lens'], results['max_diff'], marker='o', color='red')
    axes[1].set_xlabel('Context length')
    axes[1].set_ylabel('Max prediction difference')
    axes[1].set_title('Model prediction alignment')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Loss difference
    loss_diff = np.array(results['loss_with_context']) - np.array(results['loss_with_delta_w'])
    axes[2].plot(results['context_lens'], np.abs(loss_diff), marker='o', color='green')
    axes[2].set_xlabel('Context length')
    axes[2].set_ylabel('|Loss difference|')
    axes[2].set_title('Loss match quality')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path != "":
        plt.savefig(save_path, dpi=150)
    plt.show()

if __name__ == "__main__":

    model_path = "src/models/pretrained_model.pt"
    """Run all three experiments."""

    # Load or train model
    if model_path and Path(model_path).exists():
        print(f"Loading model from {model_path}")
        model = SingleLayerTransformer(D_IN, D_MLP, N_HEADS, D_K).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training model...")
        from training.train import train
        model = train(n_steps=2000, batch_size=64)
        if model_path:
            torch.save(model.state_dict(), model_path)

    results_22 = experiment_verify_theorem_22(model, n_eval=10)
    plot_theorem_verification(results_22, "src/plots/theorem_22")