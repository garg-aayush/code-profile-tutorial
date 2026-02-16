# simple run
# python train_gpt2.py

import math
import os
import time

import numpy as np
import torch
from utils.model import GPT, GPTConfig
from utils.data import get_random_batch

# -------------------------------------------------------------#
# params
# -------------------------------------------------------------#
grad_norm_clip = 1.0     # global norm gradient clipping
# data
B = 4                   # batch size
T = 1024                # sequence length
# optimizer hyperparameters
max_lr = 1.5e-3           # constant learning rate
warmup_steps = 10      # number of warmup steps for timing stabilization
steps = 50             # number of steps to profile
weight_decay = 0.1      # weight decay for optimizer
betas = (0.9, 0.95)     # betas for optimizer
# model
vocab_size = 10_000     # vocabulary size 50,000 merges + 256 byte pieces + 1 <endoftext> token -> nice number: 50,304
n_layer = 12           # number of layers
n_embd = 768           # embedding dimension
n_head = 12            # number of attention heads
# system
device = "cuda"        # device to use, "cuda" or "mps" or "cpu" (DDP only for "cuda")
seed = 42              # seed for the random number generator
device_type = "cuda"
use_compile = False    # use torch.compile to further speedup the model
max_steps = warmup_steps + steps
# -------------------------------------------------------------#
# print config keys, cool way to see the config
config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool))}
for k,v in config.items():
    print(f"{k:<20}: {v}")
# -------------------------------------------------------------#


# -------------------------------------------------------------#
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available, this script requires a GPU to run.")
print(f"Using device: {device}")

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