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
# for ddp training 
import torch.distributed as dist
from model import GPT, GPTConfig
from torch.nn.parallel import DistributedDataParallel as DDP

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
# H100 SXM: 989e12 flops (bf16)
# H100 PCIe: 756e12 flops (bf16)
# A100: 312e12 flops (bf16)
# RTX6000 Ada: 364e12 flops (bf16)
flops_promised = 989e12    # flops (bf16) promised by the gpu
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


# setup DDP training
# torchrun commands sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
# and we can use them to initialize the DDP
ddp = int(os.environ.get("RANK", -1)) != -1 # if RANK is not -1, then we are using DDP
if ddp:
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"]) # global rank
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # local rank on a single node
    ddp_world_size = int(os.environ["WORLD_SIZE"]) # total number of processes
    device = f"cuda:{ddp_local_rank}"
    device_type = 'cuda'
    torch.cuda.set_device(ddp_local_rank)
    master_process = ddp_rank == 0 # this process will do the printing, logging, checkpointing, etc.
    seed_offset = ddp_rank
    # sanity check
    print(f"Using DDP with rank {ddp_rank}, local rank {ddp_local_rank}, world size {ddp_world_size} on device {device}")
else:
    # vanilla single GPU/CPU/MPS training
    master_process = True
    ddp_rank = 0
    seed_offset = 0
    ddp_local_rank = 0
    ddp_world_size = 1

# set the seed, different for each process
torch.cuda.manual_seed(seed + seed_offset)
if device_type == "cuda":  # Fixed: only set CUDA seed if using CUDA
    torch.cuda.manual_seed(seed + seed_offset)

# use tf32
torch.set_float32_matmul_precision("high")

# get a data batch
print("Calculate gradient accumulation steps...")
assert total_batch % (B * T * ddp_world_size) == 0, f"Total batch size {total_batch} is not divisible by B*T*WORLD_SIZE={B * T * ddp_world_size}"
grad_accum_steps = total_batch // (B * T * ddp_world_size)
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
if ddp:
    # pass ddp_local_rank to the model to ensure the model is moved to the correct device
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

running_mfu = -1.0
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
        if ddp:
            # only synchronize on the final micro-step and all-reduce the loss_accum across all processes
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # use bfloat16 for the model forward pass, supported on Ampere and above
        # note since we are using bf16 and not f16, we don't need to use gradient scaler
        # As bf16 has the same range as fp32
        # Karpathy suggests to only refer to https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale down the loss by the number of gradient accumulation steps 
        # because the gradients just add up on each successive step (loss.backward())
        # and we want mean instead of sum
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        # all-reduce the loss_accum across all processes
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
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
    tokens_per_second = (B * T) * grad_accum_steps * ddp_world_size / (t1-t0)
    if master_process and (step % log_interval == 0 or last_step):
        if step > 5: # let the training loop stabilize
            mfu = raw_model.estimate_mfu(grad_accum_steps * B, dt/1000, flops_promised)
            # smooth the mfu (moving average)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"step: {step:04d}, loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/s: {tokens_per_second:.2f} | mfu: {running_mfu*100:.2f}%")
            
if ddp:
    dist.destroy_process_group()