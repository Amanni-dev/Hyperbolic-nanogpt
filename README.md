# Hyperbolic nanoGPT

A **fully hyperbolic GPT** on the Lorentz (hyperboloid) model H^d_k.

Every hidden state is a point on the hyperboloid (shape `(..., d+1)`,
`<x,x>_L = -k`).  All operations act **directly** on the manifold — there are
no `log_0`/`exp_0` round-trips through the tangent space at the origin, so
composing the layers stays genuinely hyperbolic rather than collapsing to a
Euclidean network (Chen et al., ACL 2022, "Fully Hyperbolic Neural Networks").

## Architecture

| Component | Operation |
|---|---|
| Token / position embeddings | Learned spatial coords → hyperboloid via `to_manifold` (`x₀ = √(k + ‖x_s‖²)`) |
| Embedding combination | Weighted Lorentzian centroid of token & position points (learnable `tok_w`, `pos_w`) |
| Linear layers (QKV, MLP, output) | **Lorentz-direct**: `W` acts on the full ambient `(d+1)`-vector → rotations **and** boosts, then reproject |
| RMSNorm | Spatial RMS-norm + reprojection to H^d_k (no tangent-space round-trip) |
| Activations | GELU applied to the spatial part inside `LorentzLinear`, then reprojected |
| Residual connections | 2-point Lorentzian centroid with learnable weights (`res_attn`, `res_mlp`) |
| QK similarity | `softmax(-d²_L · exp(log_scale))` — per-head learnable score scale |
| V aggregation | Weighted Lorentzian centroid over value points |
| LM head | **Lorentz MLR** — signed geodesic distance to learnable class hyperplanes (Ganea et al. 2018; Bdeir et al. ICLR 2024) |

### Why this is hyperbolic

- **No Euclidean shortcut**: linear layers act on the full ambient vector
  (including the time coordinate `x₀`), so they realise Lorentz boosts — not
  just rotations that cancel to a Euclidean matmul.
- **Curvature is active**: changing `k` measurably shifts the logits (the
  geodesic distances and MLR hyperplane distances depend on `k`), confirming
  the geometry is not a dead parameter.
- **On-manifold constraint**: after every layer, `|⟨x,x⟩_L + k| < 1e-5`
  (verified empirically on a trained checkpoint).

### Curvature

Curvature `k` is a **fixed constant** (`k_buf` buffer in the model, set from
`config.curvature`).  Earlier versions used learnable per-head curvature, but
it collapsed toward the Euclidean limit and merely acted as a logit
temperature.  Output sharpness is now decoupled via an explicit learnable
temperature (`log_tau`) in the Lorentz MLR head, and per-head score scale
(`log_scale`) in attention.

## Learnable hyperbolic parameters

| Parameter | Where | Role |
|---|---|---|
| `log_scale` | per attention head | Inverse temperature on `-d²_L` in attention scores |
| `log_tau` | LM head | Logit temperature for Lorentz MLR (decoupled from curvature) |
| `res_attn`, `res_mlp` | per block | Residual mixing weights (softplus, init ≈ 0.05) |
| `tok_w`, `pos_w` | embeddings | Centroid weights for combining token & position points |
| `z`, `a` | LM head | Class hyperplane normals & offset angles |

## Installation and Running

**Main env** (full training):

```bash
conda env create -f env.yaml
conda activate hypgpt
```

**Light env** (debug/tests, CPU):

```bash
conda create -n hypgpt-test python=3.10 -y
conda activate hypgpt-test
pip install torch transformers
```

**Debug scripts** (no GPU):

- `python debug/debug.py` — forward pass, backward, generation (Shakespeare char)

**Experiment scripts** (GPU, DDP):

- `run/shakespeare.sh` — sweep over seeds
- `run/tinystories.sh`, `run/tinystories_char.sh`
- `run/test.sh` — short run for sanity check

Run from repo root with `torchrun`, e.g.:

```bash
torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --data_path data/shakespeare_char \
    --n_layers 6 --n_heads 6 --head_dim 32 \
    --sequence_length 128 \
    --curvature 1.0 --seed 0
```

## Key Parameters

| Parameter | Description |
|---|---|
| `curvature` | Fixed curvature `k` of the hyperboloid (global) |
| `data_path` | Dataset dir: `data/shakespeare_char`, `data/tinystories`, `data/tinystories_char` |
| `n_layers`, `n_heads`, `head_dim` | Architecture (`n_embd = n_heads × head_dim`) |
| `sequence_length` | Context window |
| `batch_size`, `device_batch_size` | Global / per-device batch size |
| `num_iterations` | Training iterations |
| `max_hours` | Wall-clock budget (0 = unlimited); stops & saves before the limit |
| `gen_every`, `gen_prompt`, `gen_length` | Generation during training |
| `wte_lr`, `muon_lr`, `head_lr` | Optimizer learning rates |

## Datasets

| Dataset | Tokenizer | Vocab size | Prepare script |
|---|---|---|---|
| `shakespeare_char` | Character-level | ~65 | `data/shakespeare_char/prepare.py` |
| `tinystories` | GPT-2 BPE | 50257 | `data/tinystories/prepare.py` |
| `tinystories_char` | Character-level | ~256 | `data/tinystories_char/prepare.py` |

All datasets use the same binary format (magic `20240520`, 1024-byte header,
`uint16` tokens).  The model is tokenizer-agnostic — it only sees token IDs;
the architecture is identical across datasets.

## Inference example

Load the overfit checkpoint `hyp_gpt_overfit.pt` (6 layers, 6 heads,
`head_dim=32`, `n_embd=192`, `curvature=1.0`, trained on `shakespeare_char`).
This is an **overfit** checkpoint (sanity check that the architecture can
memorize the training set, not a generalizing model) and generate text:

```python
import torch
from custom_tokenizers.char_tokenizer import CharacterTokenizer
from model.model import GPT
from model.config import Config

tok = CharacterTokenizer.from_pretrained(save_directory="data/shakespeare_char/")
ckpt = torch.load("hyp_gpt_overfit.pt", map_location="cpu", weights_only=False)
cfg = Config(data_path="data/shakespeare_char", n_layers=6, n_heads=6, head_dim=32,
             sequence_length=128, curvature=1.0, vocab_size=tok.vocab_size)

model = GPT(cfg)
sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
model.load_state_dict(sd, strict=False)
model.eval()

ctx = tok.encode("ROMEO:", add_special_tokens=False, return_tensors="pt")
out = model.generate_text(ctx, max_length=120, temperature=0.7, top_k=40)
print(tok.decode(out[0], skip_special_tokens=True))
```

Sample output (overfit checkpoint, temperature=0.7, top_k=40):

```
ROMEO:
Thou art not death, the best of death:
Since Gentle Richmond against those that hou
will hear the horse of his soul woo
```

## Acknowledgements

- [modded-nanogpt](https://github.com/kellerjordan/nanoGPT) and [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) for the base implementation.
- [hyp-nanogpt](https://github.com/Alex2034/hyp-nanogpt.git) for the base implementation.
- [adamdivak/hyper_lm](https://github.com/adamdivak/hyper_lm) for reference hyperbolic transformer patterns.
- [HNN++](https://github.com/mil-tokyo/hyperbolic_nn_plusplus) for hyperbolic neural network primitives.

