from dataclasses import dataclass
import os
import math


@dataclass
class Config:
    print_multiplier: int = 5
    data_path: str = "data/shakespeare_char"
    input_bin: str = ""
    input_val_bin: str = ""
    sequence_length: int = 1024
    batch_size: int = 512
    device_batch_size: int = 32
    num_iterations: int = 1000
    cooldown_frac: float = 0.8
    max_hours: float = 11.5
    gen_every: int = 0
    gen_first: int = 0
    gen_length: int = 100
    gen_prompt: str = "Once "
    train_loss_every: int = 10
    val_loss_every: int = 10
    log_curv_every: int = 0
    val_tokens_frac: float = 1.
    save_every: int = 0
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 6
    head_dim: int = 0
    n_embd: int = 768
    normalization: str = "power"
    curvature: float = 1.0
    init_p: float = 2.0
    k_lr: float = 1.0
    wte_lr: float = 0.03
    muon_lr: float = 0.05
    head_lr: float = 0.03
    seed: int = 42

    def __post_init__(self):
        if self.head_dim:
            self.n_embd = self.n_heads * self.head_dim

        if self.log_curv_every == 0:
            self.log_curv_every = self.val_loss_every

        dataset_name = os.path.basename(self.data_path)

        if dataset_name in ["tinystories", "shakespeare_char", "tinystories_char"]:
            self.input_bin = f"{self.data_path}/train.bin"
            self.input_val_bin = f"{self.data_path}/val.bin"

        else:
            raise ValueError(f"Unrecognized dataset name: {dataset_name}")
