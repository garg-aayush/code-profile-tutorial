# simple run
# python train_gpt2.py
# ddp run
# torchrun --nproc_per_node=<num_gpus> train_gpt2.py
# NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=<num_gpus> train_gpt2.py # in case you get ddp error, this helped me run script on RTX6000 ada comsumer gpus

import math
import os
import time

import numpy as np
import torch
from model import GPT, GPTConfig

# -------------------------------------------------------------#
# params
# -------------------------------------------------------------#
log_interval = 1         # (steps) interval for logging
grad_norm_clip = 1.0     # global norm gradient clipping
# data
total_batch = 524288    # 2^19, ~0.5M tokens, matching GPT-3 124M model
B = 4                 # batch size
T = 1024                # sequence length
# optimizer hyperparameters
max_lr = 1.5e-3           # maximum learning rate
min_lr = max_lr * 0.1   # minimum learning rate
warmup_steps = 10      # number of warmup steps, this is from original GPT-3 paper, and is too conservative, we can even go with like 100 steps
max_steps = 100       # total number of steps, FineWeb-Edu 10B tokens (1 epoch training 10B/ 2^19)
weight_decay = 0.1      # weight decay for optimizer
betas = (0.9, 0.95)     # betas for optimizer
# model
vocab_size = 50304     # vocabulary size 50,000 merges + 256 byte pieces + 1 <endoftext> token -> nice number: 50,304
n_layer = 12           # number of layers
n_embd = 768           # embedding dimension
n_head = 12            # number of attention heads
# system
device = "cuda"        # device to use, "cuda" or "mps" or "cpu" (DDP only for "cuda")
seed = 42              # seed for the random number generator

use_compile = True    # use torch.compile to further speedup the model
# -------------------------------------------------------------#
# print config keys, cool way to see the config
config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool))}
for k,v in config.items():
    print(f"{k:<20}: {v}")
# -------------------------------------------------------------#

# -------------------------------------------------------------------------#
# data loader (random data for profiling — no real dataset needed)
# -------------------------------------------------------------------------#
seed_offset = 0

def get_random_batch(B, T, vocab_size, device):
    """Generate a random batch of token IDs"""
    x, y = (torch.randint(0, vocab_size, (B, T)) for _ in range(2))
    if "cuda" in str(device):
        x,y = map(lambda k: k.pin_memory().to(device, non_blocking=True), [x,y])
    else:
        x,y = map(lambda k: k.to(device), [x,y])
    return x, y

# -------------------------------------------------------------------------#


# -------------------------------------------------------------------------#
# helper functions
# -------------------------------------------------------------------------

# cosine decay learning-rate scheduler with warmup
def get_lr(step):
    # 1) linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) if step > max_steps, return min_lr
    if step > max_steps:
        return min_lr
    # 3) otherwise, use cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# -------------------------------------------------------------#
assert device in ["cuda", "mps", "cpu"], f"Invalid device: {device}"
if device == "cuda" and torch.cuda.is_available():
    device = "cuda"
    device_type = 'cuda'
elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    device_type = 'mps'
else:
    device = "cpu"
    device_type = 'cpu'
print(f"Using device: {device}")


# single GPU/CPU/MPS training
device_type = 'cuda' if "cuda" in device else 'cpu'

# set the seed
torch.cuda.manual_seed(seed)
if device_type == "cuda":
    torch.cuda.manual_seed(seed)

# use tf32
torch.set_float32_matmul_precision("high")

# get a data batch
print("Calculate gradient accumulation steps...")
assert total_batch % (B * T) == 0, f"Total batch size {total_batch} is not divisible by B*T={B * T}"
grad_accum_steps = total_batch // (B * T)
print(f"Total desired batch size: {total_batch}")
print(f"gradient accumulation steps: {grad_accum_steps}")

# no data loader needed — using random batches for profiling


# initialize the model
print("Initializing model...")
model_args = dict(vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd, n_head=n_head, block_size=T)
model = GPT(GPTConfig(**model_args))
model.to(device)

# setup the optimizer
print("Setting up optimizer...")
optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, betas=betas, device=device_type)

# use torch.compile to further speedup the model
if use_compile:
    print("Compiling model...")
    model = torch.compile(model)

print("Moving model to device...")

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # training
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # accumulate gradients over multiple steps
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = get_random_batch(B, T, vocab_size, device)
        # use bfloat16 for the model forward pass, supported on Ampere and above
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    
    # global norm gradient clipping at 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
    
    # determine and set the learning rate for the current step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    # torch.cuda.synchronize() to ensure the GPU finishes before timing
    torch.cuda.synchronize()
    
    # log the training loss
    t1 = time.time()
    dt = (t1 - t0) * 1000 # convert to ms
    # tokens per second is a better metric than dt because it is independent of the batch size and sequence length
    tokens_per_second = (B * T) * grad_accum_steps / (t1-t0)
    print(f"step: {step:04d}, loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/s: {tokens_per_second:.2f}")