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
B = 4                 # batch size
T = 1024                # sequence length
# optimizer hyperparameters
max_lr = 1.5e-3           # constant learning rate
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
    
    x, y = get_random_batch(B, T, vocab_size, device)
    # use bfloat16 for the model forward pass, supported on Ampere and above
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    
    loss.backward()
    
    # global norm gradient clipping at 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
    
    # constant learning rate (max_lr)
    optimizer.step()
    
    # torch.cuda.synchronize() to ensure the GPU finishes before timing
    torch.cuda.synchronize()
    
    # log the training loss
    t1 = time.time()
    dt = (t1 - t0) * 1000 # convert to ms
    # tokens per second is a better metric than dt because it is independent of the batch size and sequence length
    tokens_per_second = (B * T) / (t1-t0)
    print(f"step: {step:04d}, loss: {loss.item():.4f} | lr: {max_lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/s: {tokens_per_second:.2f}")