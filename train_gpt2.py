# simple run
# python train_gpt2.py
# python train_gpt2.py --batch_size 8 --steps 100 --mixed_precision --use_compile
# python train_gpt2.py --model_size medium

import argparse
import math
import os
import time

import numpy as np
import torch
from contextlib import nullcontext
from utils.model import GPT, GPTConfig
from utils.data import get_random_batch
from utils.constants import *


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GPT-2 training benchmark script")

    # data
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--seq_len", type=int, default=1024, help="sequence length")

    # optimizer hyperparameters
    parser.add_argument("--warmup_steps", type=int, default=10, help="number of warmup steps for timing stabilization")
    parser.add_argument("--steps", type=int, default=50, help="number of steps to profile")

    # model
    parser.add_argument("--model_size", type=str, default="small", choices=MODEL_CONFIGS.keys(),
                        help=f"model size configuration, one of: {list(MODEL_CONFIGS.keys())}")

    # system
    parser.add_argument("--seed", type=int, default=42, help="seed for the random number generator")
    parser.add_argument("--use_compile", action="store_true", help="use torch.compile to further speedup the model")
    parser.add_argument("--mixed_precision", action="store_true", help="use bfloat16 mixed precision")
    parser.add_argument("--use_tf32", action="store_true", help="use tf32 precision")

    args = parser.parse_args()
    return args


def benchmark(args):
    """Benchmark the training loop with the given args."""
    # get model config
    model_cfg = MODEL_CONFIGS[args.model_size]
    vocab_size = model_cfg["vocab_size"]

    # derived values
    betas = (BETA1, BETA2)
    device_type = "cuda" if "cuda" in DEVICE else DEVICE
    max_steps = args.warmup_steps + args.steps

    # -------------------------------------------------------------#
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, this script requires a GPU to run.")
    print(f"Using device: {DEVICE}")

    # set the seed
    torch.cuda.manual_seed(args.seed)

    # use tf32
    if args.use_tf32:
        torch.set_float32_matmul_precision("high")

    # initialize the model
    print(f"Initializing model ({args.model_size})...")
    gpt_config = GPTConfig(**model_cfg, block_size=args.seq_len)
    print(gpt_config)
    model = GPT(gpt_config)
    model.to(DEVICE)

    # setup the optimizer
    print("Setting up optimizer...")
    optimizer = model.configure_optimizers(weight_decay=WEIGHT_DECAY, learning_rate=MAX_LR, betas=betas, device=device_type)

    # use torch.compile to further speedup the model
    if args.use_compile:
        print("Compiling model...")
        model = torch.compile(model)

    print("Moving model to device...")

    # mixed precision context
    print(f"Using mixed precision: {args.mixed_precision}")
    ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if args.mixed_precision else nullcontext()

    model.train()
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        optimizer.zero_grad(set_to_none=True)

        x, y = get_random_batch(args.batch_size, args.seq_len, vocab_size, DEVICE)

        with ctx:
            _, loss = model(x, y)
        loss.backward()

        # global norm gradient clipping at 1.0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)

        # constant learning rate (max_lr)
        optimizer.step()

        # torch.cuda.synchronize() to ensure the GPU finishes before timing
        torch.cuda.synchronize()

        # log the training loss
        t1 = time.time()
        dt = (t1 - t0) * 1000 # convert to ms
        tokens_per_second = (args.batch_size * args.seq_len) / (t1-t0)
        print(f"step: {step:04d}, loss: {loss.item():.4f} | dt: {dt:.2f} ms | tok/s: {tokens_per_second:.2f}")


def main():
    """Main entry point."""
    args = parse_args()

    # print config
    for k, v in vars(args).items():
        print(f"{k:<20}: {v}")

    benchmark(args)


if __name__ == "__main__":
    main()