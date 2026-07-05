"""
Load and test a trained hyperbolic GPT checkpoint.
Works on CPU. Usage:
    PYTHONPATH=. python3 debug/test_checkpoint.py [path/to/checkpoint.pt]
"""

import sys
import math
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
from custom_tokenizers.char_tokenizer import CharacterTokenizer
from model.model import GPT
from model.config import Config

# --- load checkpoint -------------------------------------------------------
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "hyp_gpt_final.pt"
print(f"Loading checkpoint: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
saved_cfg = ckpt.get("config", {})
print(f"Step: {ckpt['step']}  |  Best val: {ckpt.get('best_val', '?')}")

# --- build model from saved config -----------------------------------------
config = Config(data_path="data/shakespeare_char")
config.n_layers = saved_cfg.get("n_layers", 8)
config.n_heads = saved_cfg.get("n_heads", 4)
config.head_dim = saved_cfg.get("head_dim", 16)
config.sequence_length = saved_cfg.get("sequence_length", 256)
config.n_embd = config.n_heads * config.head_dim
config.normalization = saved_cfg.get("normalization", "power")
config.curvature = saved_cfg.get("curvature", 1.0)
config.k_lr = saved_cfg.get("k_lr", 1.0)
config.init_p = saved_cfg.get("init_p", 2.0)

tokenizer = CharacterTokenizer.from_pretrained(save_directory="data/shakespeare_char/")
config.vocab_size = tokenizer.vocab_size

model = GPT(config)
print(f"Model: {model.model_size()} ({sum(p.numel() for p in model.parameters()):,} params)")

# --- strip _orig_mod. prefix from compiled state dict ----------------------
state_dict = ckpt["model"]
clean_sd = {}
for k, v in state_dict.items():
    clean_k = k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k
    clean_sd[clean_k] = v

missing, unexpected = model.load_state_dict(clean_sd, strict=False)
if missing:
    print(f"Missing keys: {missing}")
if unexpected:
    print(f"Unexpected keys: {unexpected}")
print("Model loaded successfully!\n")

# --- curvature stats -------------------------------------------------------
k_global = model.k_global.item()
print(f"Global curvature k = {k_global:.4f}")
print("Per-layer head curvatures:")
for i, blk in enumerate(model.transformer.h):
    if hasattr(blk.attn, "log_c"):
        cs = torch.exp(blk.attn.log_c.detach()).cpu().flatten()
        cs_str = "  ".join(f"{c:.4f}" for c in cs)
        print(f"  layer {i}: [{cs_str}]")
print()

# --- generation ------------------------------------------------------------
model.eval()

prompts = [
    "ROMEO:",
    "JULIET:",
    "KING HENRY:",
    "The ",
    "To be, or not to be",
    "MENENIUS:",
]

print("=" * 70)
print("Generation samples (temperature=0.7, top_k=30)")
print("=" * 70)

for prompt in prompts:
    context = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate_text(
            context, max_length=150, temperature=0.7, top_k=30
        )
    decoded = tokenizer.decode(generated[0].tolist())
    print(f"\n{decoded}")
    print("-" * 70)

# --- also try with greedy (temperature=0.01) for most likely tokens --------
print("\n" + "=" * 70)
print("Greedy samples (temperature=0.01)")
print("=" * 70)

for prompt in ["ROMEO:", "The "]:
    context = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate_text(
            context, max_length=100, temperature=0.01, top_k=1
        )
    decoded = tokenizer.decode(generated[0].tolist())
    print(f"\n{decoded}")
    print("-" * 70)
