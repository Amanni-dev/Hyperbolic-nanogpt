"""
Debug script for the fully hyperbolic GPT.
Tests forward pass, backward pass, and generation on shakespeare_char.
No GPU required — runs on CPU with a tiny model.
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

# Initialize tokenizer
tokenizer = CharacterTokenizer.from_pretrained(save_directory="data/shakespeare_char/")

# Setup config — tiny model for CPU debugging
config = Config(data_path="data/shakespeare_char")
config.n_layers = 2
config.n_heads = 2
config.head_dim = 8
config.sequence_length = 32
config.n_embd = config.n_heads * config.head_dim  # 16
config.batch_size = 1
config.curvature = 1.0
config.k_lr = 1.0
config.normalization = "power"
config.vocab_size = tokenizer.vocab_size

# Create model
model = GPT(config)
print(f"Model size: {model.model_size()}")
print(f"Global curvature k = {model.k.item():.4f}")

# Create dummy input
prompt = "Once upon a time in a"
input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
idx = input_ids[:, :-1]
targets = input_ids[:, 1:].clone()

# Forward pass
logits, loss = model(idx, targets=targets)
print("\nForward pass successful!")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss.item():.4f}")

# Backward pass
loss.backward()
print("Backward pass successful!")

# Check that gradients exist for key parameters
grad_info = []
for name, p in model.named_parameters():
    if p.grad is not None:
        gn = p.grad.norm().item()
        grad_info.append(f"  {name}: grad_norm={gn:.4g}")
    else:
        grad_info.append(f"  {name}: NO GRAD")
print("\nGradient info (first 20):")
for line in grad_info[:20]:
    print(line)

# Generation test
model.eval()
gen_prompt = "The "
gen_context = tokenizer.encode(gen_prompt, add_special_tokens=False, return_tensors="pt")
generated = model.generate_text(gen_context, max_length=20, temperature=0.8, top_k=40)
decoded = tokenizer.decode(generated[0].tolist())
print("\nGeneration successful!")
print(f"Prompt: {gen_prompt!r}")
print(f"Generated: {decoded!r}")
