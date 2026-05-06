# Learning without training: The implicit dynamics of in-context learning

A PyTorch replication of the experiments from:

> **Learning without training: The implicit dynamics of in-context learning**
> Benoit Dherin, Michael Munn, Hanna Mazzawi, Michael Wunder, Javier Gonzalvo
> Google Research, 2025
> [arXiv:2507.16003](https://arxiv.org/abs/2507.16003)

---

## Overview

This repository reproduces the core experiments of Dherin et al. (2025), which investigate the mechanism behind in-context learning (ICL) in transformers. The paper's central result is **Theorem 2.2**: the effect of a context `C` on a transformer block's output can be expressed exactly as a rank-1 update of the MLP's first-layer weight matrix:

```
TW(C, x)  ==  TW+ΔW(x)

where  ΔW = (W · δA_x(C)) · A(x)ᵀ / ‖A(x)‖²
```

This means that instead of passing context tokens through the model, you can equivalently modify the MLP weights by a rank-1 matrix `ΔW` computed from the attention output — and get the exact same prediction.

---

## Experiments Replicated

### 1. Training
A single-layer decoder-only transformer is trained on linear regression tasks. Each prompt is a sequence of in-context input-output pairs `(xi, f(xi))` where `f(x) = ωᵀx`, followed by a query token. The model learns to predict `f(xquery)` purely from the in-context examples, with no access to `ω`.

### 2. Theorem 2.2 Verification
After training, the model weights are frozen. For each test prompt we compute `ΔW` and verify that:
- The prediction with full context `TW(C, x)` 
- The prediction with only the query token but modified weights `TW+ΔW(x)`

are identical up to numerical precision (~1e-6).

### 3. Implicit Learning Dynamics
As context tokens are consumed one by one, the sequence of implicit weight updates `‖ΔW_{i+1} − ΔW_i‖` is tracked. This converges to zero as the full context is processed, mirroring the behavior of a converging gradient descent optimizer (Proposition 3.1 of the paper).

---

## Project Structure

```
InContext-Sequence/
├── src/
│   ├── transformer/
│   │   └── transformer.py       # Single-layer transformer architecture
│   ├── training/
│   │   └── train.py             # Training and test loop
│   └── experiments/
│       └── verify_theorem.py    # Theorem 2.2 verification + plots
├── requirements.txt
└── README.md
```

---

## Architecture

The transformer follows Section 4.1 of the paper exactly:

| Hyperparameter | Value |
|---|---|
| Feature dimension `d` | 2 |
| In-context examples `N` | 100 |
| MLP hidden dim `D_MLP` | 128 |
| Attention output dim `D_MODEL` | 32 |
| Attention heads `h` | 8 |
| Per-head dim `D_K` | 4 |
| Token dim `D_IN` | 3 (= d + 1) |

- Decoder-only with causal mask
- No skip connections, no LayerNorm, no positional encoding

---

## Installation

```bash
git clone https://github.com/Magalop-bit/InContext-Sequence.git
cd InContext-Sequence
pip install -r requirements.txt
```

---

## Usage

**Train the model:**
```bash
python src/training/train.py
```

**Verify Theorem 2.2:**
```bash
python src/experiment/experiment.py
```

A pretrained model will be saved to `src/models/pretrained_model.pt` and reused automatically on subsequent runs.

---

## Expected Results

After sufficient training, the verification should produce:

```
Mean loss with context:  ~0.13
Mean loss with ΔW:       ~0.13
Mean prediction diff:    ~1e-6
```

The near-zero prediction difference confirms Theorem 2.2 — the context is exactly captured by the rank-1 weight update.

---

## Citation

```bibtex
@article{dherin2025learning,
  title   = {Learning without training: The implicit dynamics of in-context learning},
  author  = {Dherin, Benoit and Munn, Michael and Mazzawi, Hanna and Wunder, Michael and Gonzalvo, Javier},
  journal = {arXiv preprint arXiv:2507.16003},
  year    = {2025}
}
```
