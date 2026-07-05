"""
CPU training of the fully hyperbolic GPT on shakespeare_char.

This is a self-contained training loop (no DDP, no GPU) used to verify
that the hyperbolic model learns and to inspect curvature dynamics.
"""

import sys
import math
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import torch
import torch.nn.functional as F
from custom_tokenizers.char_tokenizer import CharacterTokenizer
from model.model import GPT
from model.config import Config

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
tokenizer = CharacterTokenizer.from_pretrained(save_directory="data/shakespeare_char/")

config = Config(data_path="data/shakespeare_char")
config.n_layers = 4
config.n_heads = 4
config.head_dim = 16
config.sequence_length = 128
config.n_embd = config.n_heads * config.head_dim  # 64
config.curvature = 1.0
config.k_lr = 1.0
config.normalization = "power"
config.init_p = 2.0
config.vocab_size = tokenizer.vocab_size

model = GPT(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model size: {model.model_size()}")
print(f"Total params: {total_params:,}")
print(f"Architecture: {config.n_layers}L {config.n_heads}H d={config.n_embd} "
      f"seq={config.sequence_length}")
print(f"Hyperbolic: normalization={config.normalization} "
      f"curvature={config.curvature} k_lr={config.k_lr} p={config.init_p}")
print()

# Load training data (skip 256*4 byte header)
with open("data/shakespeare_char/train.bin", "rb") as f:
    f.read(256 * 4)
    train_data = torch.tensor(
        np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
    ).long()

with open("data/shakespeare_char/val.bin", "rb") as f:
    f.read(256 * 4)
    val_data = torch.tensor(
        np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
    ).long()

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens:   {len(val_data):,}")
print()

# ---------------------------------------------------------------------------
# Optimizer — separate parameter groups (mirrors train_gpt2.py logic)
# ---------------------------------------------------------------------------
curv_params = []
matrix_params = []
embed_params = []
other_params = []

for name, p in model.named_parameters():
    if name.endswith('.log_c') or name == 'log_k_global':
        curv_params.append(p)
    elif name in ('transformer.wte.weight', 'transformer.wpe.weight'):
        embed_params.append(p)
    elif p.ndim == 2:
        matrix_params.append(p)
    else:
        other_params.append(p)

lr_matrix = 3e-3
lr_embed = 3e-3
lr_curv = 1.0
lr_other = 3e-3

optimizer = torch.optim.Adam(
    matrix_params + embed_params + other_params,
    lr=lr_matrix, betas=(0.9, 0.95), eps=1e-10,
)
optimizer_curv = torch.optim.SGD(curv_params, lr=lr_curv, momentum=0.0)

print("Parameter groups:")
print(f"  curvature:  {sum(p.numel() for p in curv_params):>8,}  (SGD lr={lr_curv})")
print(f"  matrix:     {sum(p.numel() for p in matrix_params):>8,}  (Adam lr={lr_matrix})")
print(f"  embedding:  {sum(p.numel() for p in embed_params):>8,}  (Adam lr={lr_embed})")
print(f"  other:      {sum(p.numel() for p in other_params):>8,}  (Adam lr={lr_other})")
print()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
B, T = 12, config.sequence_length
NUM_STEPS = 2000
WARMUP = 100
LR_DECAY = 0.1  # final lr = peak * LR_DECAY

def get_lr(step):
    """Linear warmup then cosine decay."""
    if step < WARMUP:
        return step / WARMUP
    progress = (step - WARMUP) / max(1, NUM_STEPS - WARMUP)
    return LR_DECAY + (1 - LR_DECAY) * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

def clamp_curvature(m):
    with torch.no_grad():
        m.log_k_global.clamp_(min=math.log(1e-5), max=math.log(1e3))
        for blk in m.transformer.h:
            if hasattr(blk.attn, 'log_c'):
                blk.attn.log_c.clamp_(min=math.log(1e-5), max=math.log(1e3))
            if hasattr(blk.attn, 'p') and isinstance(blk.attn.p, torch.nn.Parameter):
                blk.attn.p.clamp_(min=1e-2, max=1e2)

@torch.no_grad()
def evaluate(m, data, num_batches=20):
    m.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        ix = torch.randint(0, len(data) - T - 1, (B,))
        x = torch.stack([data[i:i + T + 1] for i in ix])
        _, loss = m(x[:, :-1].contiguous(), targets=x[:, 1:].contiguous())
        total_loss += loss.item()
    m.train()
    return total_loss / num_batches

def curvature_stats(m):
    k_global = m.k_global.item()
    head_curvs = []
    for blk in m.transformer.h:
        if hasattr(blk.attn, 'log_c'):
            head_curvs.append(torch.exp(blk.attn.log_c.detach()).cpu().flatten())
    if head_curvs:
        all_c = torch.cat(head_curvs)
        return (k_global, all_c.mean().item(), all_c.std(unbiased=False).item(),
                all_c.min().item(), all_c.max().item())
    return (k_global, 0, 0, 0, 0)

print(f"Training {NUM_STEPS} steps | batch={B} | seq={T} | "
      f"tokens/step={B*T:,}")
print("=" * 80)

train_losses = []
val_losses = []
t0 = time.time()

model.train()
for step in range(NUM_STEPS):
    # sample batch
    ix = torch.randint(0, len(train_data) - T - 1, (B,))
    x = torch.stack([train_data[i:i + T + 1] for i in ix])
    idx = x[:, :-1].contiguous()
    targets = x[:, 1:].contiguous()

    # forward
    _, loss = model(idx, targets=targets)

    # backward
    optimizer.zero_grad()
    optimizer_curv.zero_grad()
    loss.backward()
    loss_f = loss.item()  # already per-token (mean over B*T)

    clamp_curvature(model)

    optimizer.step()
    optimizer_curv.step()
    scheduler.step()

    train_losses.append(loss_f)

    # logging
    if step % 200 == 0 or step == NUM_STEPS - 1:
        val_loss = evaluate(model, val_data, num_batches=10)
        val_losses.append(val_loss)
        kg, cm, cs, cmin, cmax = curvature_stats(model)
        elapsed = time.time() - t0
        avg_train = sum(train_losses[-100:]) / min(100, len(train_losses))
        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"step {step:4d} | lr {lr_now:.4f} | "
            f"train {avg_train:.4f} | val {val_loss:.4f} | "
            f"k_glob {kg:.3f} | k_head {cm:.3f}±{cs:.3f} "
            f"[{cmin:.3f},{cmax:.3f}] | "
            f"{elapsed:.1f}s"
        )

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("=" * 80)
initial_train = sum(train_losses[:20]) / 20
final_train = sum(train_losses[-20:]) / 20
initial_val = val_losses[0] if val_losses else 0
final_val = val_losses[-1] if val_losses else 0

print(f"\nSummary:")
print(f"  Train loss: {initial_train:.4f} -> {final_train:.4f}  (Δ={initial_train - final_train:.4f})")
print(f"  Val loss:   {initial_val:.4f} -> {final_val:.4f}  (Δ={initial_val - final_val:.4f})")
print(f"  Total time: {time.time() - t0:.1f}s")

kg, cm, cs, cmin, cmax = curvature_stats(model)
print(f"\nFinal curvature:")
print(f"  Global k = {kg:.4f}")
print(f"  Per-head  = {cm:.4f} ± {cs:.4f}  (min={cmin:.4f}, max={cmax:.4f})")

# Per-layer head curvature breakdown
print(f"\nPer-layer head curvatures:")
for i, blk in enumerate(model.transformer.h):
    if hasattr(blk.attn, 'log_c'):
        cs_layer = torch.exp(blk.attn.log_c.detach()).cpu().flatten()
        cs_str = " ".join(f"{c:.3f}" for c in cs_layer)
        print(f"  layer {i}: [{cs_str}]")

# ---------------------------------------------------------------------------
# Save model checkpoint
# ---------------------------------------------------------------------------
import os
ckpt_dir = "debug/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "hyp_gpt_cpu.pt")
torch.save({
    "model": model.state_dict(),
    "config": vars(config),
    "final_val_loss": final_val,
    "final_train_loss": final_train,
}, ckpt_path)
print(f"\nModel saved to: {ckpt_path}")

# ---------------------------------------------------------------------------
# Generation samples
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Generation samples:")
print("=" * 80)

model.eval()
prompts = ["ROMEO:", "JULIET:", "The ", "KING HENRY:"]
for prompt in prompts:
    context = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate_text(
            context, max_length=80, temperature=0.7, top_k=30
        )
    decoded = tokenizer.decode(generated[0].tolist())
    print(f"\n{decoded}")
