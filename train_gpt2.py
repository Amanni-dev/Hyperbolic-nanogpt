import os
import math
import time
import random
import datetime
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2TokenizerFast  # type: ignore

from custom_tokenizers.char_tokenizer import CharacterTokenizer
from model.model import GPT
from model.config import Config
from utils.muon import Muon
from utils.loader import DistributedDataLoader
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="data/shakespeare_char")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device_batch_size", type=int, default=32)
parser.add_argument("--num_iterations", type=int, default=4)
parser.add_argument("--gen_every", type=int, default=0)
parser.add_argument("--gen_prompt", type=str, default="Once ")
parser.add_argument("--gen_first", type=int, default=0)
parser.add_argument("--gen_length", type=int, default=200)
parser.add_argument("--train_loss_every", type=int, default=2)
parser.add_argument("--val_loss_every", type=int, default=2)
parser.add_argument("--log_curv_every", type=int, default=0)
parser.add_argument("--save_every", type=int, default=2000)
parser.add_argument("--head_dim", type=int, default=16)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--n_layers", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sequence_length", type=int, default=128)
parser.add_argument("--k_lr", type=float, default=1.0)
parser.add_argument("--curvature", type=float, default=1.0)
parser.add_argument("--normalization", type=str, default="power",
                    help="'power', 'exp', or 'learnable'")
parser.add_argument("--init_p", type=float, default=2.,
                    help="Initial power parameter for hyperbolic attention")
parser.add_argument("--wte_lr", type=float, default=0.03,
                    help="Learning rate for Adam (embeddings + norms/biases/scalars)")
parser.add_argument("--muon_lr", type=float, default=0.05,
                    help="Learning rate for Muon (hidden LorentzLinear matrices)")
parser.add_argument("--head_lr", type=float, default=0.03,
                    help="Learning rate for Adam (Lorentz MLR head params)")
parser.add_argument("--print_multiplier", type=int, default=5)
parser.add_argument("--max_hours", type=float, default=11.5,
                    help="Wall-clock training budget in hours (0 = unlimited); "
                         "stops and saves the checkpoint before Kaggle's 12h kill.")

args = parser.parse_args()
config = Config(**vars(args))

train_deadline = (time.time() + config.max_hours * 3600) if config.max_hours > 0 \
    else float('inf')

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

char_datasets = {"shakespeare_char", "tinystories_char"}
gpt2_datasets = {"tinystories"}

dataset_name = os.path.basename(config.data_path)

if dataset_name in char_datasets:
    tokenizer = CharacterTokenizer.from_pretrained(save_directory=config.data_path)
    config.vocab_size = tokenizer.vocab_size
elif dataset_name in gpt2_datasets:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size
else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def encode_text(tokenizer, text, device):
    return tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    ).to(device)


def decode_tokens(tokenizer, tokens):
    if "char" in config.data_path:
        return ''.join(tokenizer.convert_ids_to_tokens(tokens.cpu().tolist()))
    return tokenizer.decode(tokens.cpu().tolist(), skip_special_tokens=True)


assert torch.cuda.is_available(), "CUDA is required for DDP but not available."
try:
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
except KeyError as e:
    raise RuntimeError(f"Missing environment variable for DDP: {e}")
dist.init_process_group(backend='nccl')
device = torch.device(f'cuda:{ddp_local_rank}')
torch.cuda.set_device(device)

print(f"[Rank {ddp_rank}] Using device: {device}")

master_process = (ddp_rank == 0)
B, T = config.device_batch_size, config.sequence_length

assert config.batch_size % (B * ddp_world_size) == 0, "global batch_size must be \
    divisible by (device_batch_size * world_size)."
train_accumulation_steps = config.batch_size // (B * ddp_world_size)
tokens_per_iter = config.batch_size * config.sequence_length

train_loader = DistributedDataLoader(
    config.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(
    config.input_val_bin, B, T, ddp_rank, ddp_world_size)
val_steps = int(config.val_tokens_frac * val_loader.ntok_total) \
    // (B * T * ddp_world_size)

if master_process:
    print(
        f"Training DataLoader: {train_loader.ntok_total / 1e6:.2f}M tokens "
        f"across {len(train_loader.files)} files."
    )
    print(
        f"Validation DataLoader: {val_loader.ntok_total / 1e6:.2f}M tokens "
        f"across {len(val_loader.files)} files."
    )
    print(f"Tokenizer vocab size: {config.vocab_size}")

x, y = train_loader.next_batch()

model = GPT(config)
model = model.to(device)

matrix_params = []
embed_params = []
head_params = []
other_params = []
curv_params = []

for name, p in model.named_parameters():
    if name in ('transformer.wte.weight', 'transformer.wpe.weight'):
        embed_params.append(p)
    elif name.startswith('lm_head'):
        head_params.append(p)
    elif p.ndim == 2:
        matrix_params.append(p)
    else:
        other_params.append(p)

optimizer_muon = Muon(matrix_params, lr=config.muon_lr, momentum=0.95)
optimizer_adam = torch.optim.Adam(
    [
        {'params': embed_params, 'lr': config.wte_lr},
        {'params': head_params, 'lr': config.head_lr},
        {'params': other_params, 'lr': config.wte_lr},
    ],
    betas=(0.8, 0.95), eps=1e-10, fused=True,
)

optimizers = [optimizer_muon, optimizer_adam]

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

model = torch.compile(model)

end_event.record()
torch.cuda.synchronize()

compile_time = start_event.elapsed_time(end_event)
print(f"Model compiled in {compile_time:.1f}ms")

model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float32)

init_lr = 1.0
end_lr = 0.1


def get_lr(it):
    t = max(0, min(1, 1 - it / config.num_iterations))
    w = min(t / config.cooldown_frac, 1.0)
    return w * init_lr + (1 - w) * end_lr


schedulers = [
    torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers
]


def hyp_stats(model):
    k = model.k.item()
    scales, res = [], []
    for blk in model.transformer.h:
        scales.append(torch.exp(blk.attn.log_scale.detach()).cpu().reshape(-1))
        res.append(F.softplus(blk.res_attn.detach()).cpu().reshape(-1))
        res.append(F.softplus(blk.res_mlp.detach()).cpu().reshape(-1))
    scales = torch.cat(scales)
    res = torch.cat(res)
    tau = torch.exp(model.lm_head.log_tau.detach()).cpu().item()
    return (
        f"k(fixed)={k:.3g} | score_scale {scales.mean():.3g}±{scales.std(unbiased=False):.2g} "
        f"| res_w {res.mean():.3g}±{res.std(unbiased=False):.2g} | head_tau {tau:.3g}"
    )


def n_params(group):
    return sum(p.numel() for p in group)


def grad_norm(params, norm_type=2):
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    device = params[0].grad.device
    norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in params]
        ), norm_type)
    return norm.item()


def log_hyp(model, step):
    writer.add_scalar("Hyp/curvature_fixed", model.k.item(), step)
    writer.add_scalar("Hyp/head_tau", torch.exp(model.lm_head.log_tau).item(), step)
    for i, blk in enumerate(model.transformer.h):
        sc = torch.exp(blk.attn.log_scale.detach()).cpu().flatten()
        for j, c in enumerate(sc):
            writer.add_scalar(f"ScoreScale/layer_{i}/head_{j}", c.item(), step)


if master_process:
    model_size = raw_model.model_size()
    print("\n=== Model (Fully Hyperbolic) ===")
    print(f"Model Size:    {model_size}\n")
    print("Parameter groups:")
    print(f"mat(Muon):{n_params(matrix_params):,} | "
          f"embed:{n_params(embed_params):,} | "
          f"head:{n_params(head_params):,} | "
          f"other:{n_params(other_params):,}\n")
    print(f"Data Path:            {config.data_path}")
    print(f"Sequence Length:      {config.sequence_length}")
    print(f"Total Tokens:      {config.num_iterations * tokens_per_iter:,}")
    print(f"Batch Size (global):  {config.batch_size}")
    print(f"Batch Size (device):  {config.device_batch_size}")
    print(f"n_layers:              {config.n_layers}")
    print(f"n_heads:               {config.n_heads}")
    print(f"head_dim:             {config.head_dim}")
    print(f"n_embd:               {config.n_embd}")
    print("\n=== Hyperbolic (fully-Lorentz) ===")
    print(f"Curvature (FIXED):    {config.curvature}  (sectional -1/k)")
    print(f"Attention:            softmax(-d^2_L * scale) + Lorentzian centroid")
    print(f"Head:                 Lorentz MLR (hyperplane, asinh)")
    print(f"Seed:                 {config.seed}")
    print("==============================\n")

if master_process:
    def create_run_id(config, dataset_name, timestamp):
        dataset_aliases = {
            'shakespeare_char': 'sh',
            'tinystories_char': 'tsc',
            'tinystories': 'ts',
        }
        norm_aliases = {'power': 'pow', 'exp': 'exp', 'learnable': 'lrn'}
        date = timestamp.strftime('%m.%d')
        seconds_since_midnight = (
            timestamp - timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        ).seconds
        norm = norm_aliases[config.normalization]
        hyp_params = ""
        if config.k_lr:
            hyp_params += f"_lr{config.k_lr:.0f}"
        else:
            hyp_params += f"_c{config.curvature:.0f}"
        run_id = (
            f"{seconds_since_midnight:05d}_"
            f"{dataset_aliases[dataset_name]}_"
            f"hyp_{norm}{hyp_params}_"
            f"{model_size}_"
            f"s{config.seed}"
        )
        return date, run_id

    now = datetime.datetime.now()
    date, run_id = create_run_id(config, dataset_name, now)
    logdir = f'tensorboard_runs/{date}/{run_id}/'
    ckpt_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "tensorboard_logs"), exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Logs for this run will be stored in: {logdir}")
    print("Writing logs to: " + os.path.join(logdir, "tensorboard_logs"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, "tensorboard_logs"))

    config_path = os.path.join(logdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    def pretty_json(hp):
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    writer.add_text("run_params", pretty_json(vars(args)))


total_start = torch.cuda.Event(enable_timing=True)
interval_start = torch.cuda.Event(enable_timing=True)
interval_end = torch.cuda.Event(enable_timing=True)
step_estimates = []
total_start.record()
interval_start.record()

train_loss_accum = 0.0
train_log_count = 0

val_loss_accum = 0.0
val_log_count = 0

best_val_loss = float('inf')

train_loader.reset()
for step in range(config.num_iterations + 1):
    time_up = time.time() > train_deadline
    last_step = (step == config.num_iterations) or time_up
    if (last_step or (config.val_loss_every > 0
                      and step % config.val_loss_every == 0)):
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx:
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        val_loss_accum += val_loss
        val_log_count += 1

    if last_step:
        if time_up and master_process:
            print(f"\n[time budget] {config.max_hours:.2f}h reached at step {step}; "
                  f"stopping to save the checkpoint before Kaggle's 12h limit.")
        break

    model.train()
    for i in range(1, train_accumulation_steps + 1):
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        x, y = train_loader.next_batch()
        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        p.grad /= train_accumulation_steps

    if master_process and step % config.train_loss_every == 0:
        gn_matrix = grad_norm(matrix_params)
        gn_embed = grad_norm(embed_params)
        gn_head = grad_norm(head_params)
        gn_other = grad_norm(other_params)

        writer.add_scalar('grad_norm/matrix', gn_matrix, step)
        writer.add_scalar('grad_norm/embed', gn_embed, step)
        writer.add_scalar('grad_norm/head', gn_head, step)
        writer.add_scalar('grad_norm/other', gn_other, step)
        if step % (config.print_multiplier * config.train_loss_every) == 0:
            grads_string = (
                f"Grad norms: matrix={gn_matrix:.3g} | "
                f"embed={gn_embed:.3g} | "
                f"head={gn_head:.3g} | "
                f"other={gn_other:.3g} | "
            )

    all_grad_params = [p for p in model.parameters() if p.grad is not None]
    torch.nn.utils.clip_grad_norm_(all_grad_params, max_norm=1.0)

    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()

    if torch.isnan(train_loss) or torch.isinf(train_loss):
        if master_process:
            print(f"WARNING: NaN/Inf loss at step {step}, zeroing grads and skipping")
        model.zero_grad(set_to_none=True)
        continue

    model.zero_grad(set_to_none=True)
    train_loss_accum += train_loss.item()
    train_log_count += 1

    if master_process and step % config.train_loss_every == 0:
        interval_end.record()
        torch.cuda.synchronize()

        interval_time_ms = interval_start.elapsed_time(interval_end)
        step_estimates.append(interval_time_ms / config.train_loss_every)
        if len(step_estimates) >= 10:
            avg_time_per_step = sum(step_estimates[-10:]) / 10.
        elif len(step_estimates):
            avg_time_per_step = sum(step_estimates) / len(step_estimates)
        else:
            avg_time_per_step = np.nan

        estimated_total_time = avg_time_per_step * \
            (config.num_iterations - step) / 1e3
        avg_train_loss = train_loss_accum / max(1, train_log_count)
        avg_val_loss = val_loss_accum / max(1, val_log_count)

        tokens_seen = step * tokens_per_iter
        writer.add_scalar('Loss/Train', avg_train_loss, tokens_seen)
        writer.add_scalar('Loss/Validation', avg_val_loss, tokens_seen)
        print(
            f"step {step} ({interval_time_ms:.0f}ms):\t"
            f"{tokens_seen/1e6:.1f}M tokens seen,\t"
            f"train loss = {avg_train_loss:.4f},\t"
            f"val loss = {avg_val_loss:.4f},\t"
            f"ETA = {estimated_total_time:.0f}s"
        )
        if step % (config.print_multiplier * config.train_loss_every) == 0:
            print(f"\n{hyp_stats(raw_model)}\n{grads_string}\n")
        if config.log_curv_every and step % config.log_curv_every == 0:
            log_hyp(raw_model, step)

        train_loss_accum = 0.0
        train_log_count = 0
        val_loss_accum = 0.0
        val_log_count = 0

        if config.save_every and (step % config.save_every == 0 or last_step):
            ckpt = dict(step=step,
                        model=raw_model.state_dict(),
                        optimizers=[opt.state_dict() for opt in optimizers],
                        config=vars(args),
                        best_val=min(best_val_loss, avg_val_loss))
            path = os.path.join(ckpt_dir, f"hypnano_{step:05d}.pt")
            torch.save(ckpt, path)
            best_val_loss = min(best_val_loss, avg_val_loss)
            print(f"  ↳ checkpoint saved: {path}")

        if config.gen_every and master_process and \
           (step % config.gen_every == 0) and (config.gen_first + step):
            context = encode_text(tokenizer, config.gen_prompt, device)
            generated_tokens = raw_model.generate_text(
                context,
                max_length=config.gen_length,
                temperature=1.0,
                top_k=50
            )
            generated_text = decode_tokens(tokenizer, generated_tokens[0])
            writer.add_text(
                f"Generated_Text/Step_{step}", generated_text, step)
            print(f"\nGenerated Text: \n{generated_text}\n")
        interval_start.record()

if master_process:
    total_end_event = torch.cuda.Event(enable_timing=True)
    total_end_event.record()
    torch.cuda.synchronize()

    total_time_s = total_start.elapsed_time(total_end_event) / 1e3
    time_msg = f"Total training time: {total_time_s:.2f}s"
    print(time_msg)
    writer.add_text("Time", time_msg, step)
    mem_msg = (
        f"Peak memory consumption: "
        f"{torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )
    print(mem_msg)
    writer.add_text("GPU", mem_msg, step)

    final_ckpt_path = os.path.join(ckpt_dir, "hypnano.pt")
    torch.save({
        "step": step,
        "model": raw_model.state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "config": vars(args),
        "best_val": best_val_loss,
    }, final_ckpt_path)
    print(f"Final model saved to: {final_ckpt_path}")

if master_process:
    writer.close()

dist.destroy_process_group()
