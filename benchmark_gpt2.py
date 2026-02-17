# simple run
# python train_gpt2.py
# python train_gpt2.py --batch_size 8 --steps 100 --mixed_precision --use_compile
# python train_gpt2.py --model_size medium

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
from contextlib import nullcontext
from utils.model import GPT, GPTConfig
from utils.data import get_random_batch
from utils.constants import *
from utils.helper import build_config_metadata
import timeit


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GPT-2 training benchmark script")

    # data
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--seq_len", type=int, default=1024, help="sequence length")

    # optimizer hyperparameters
    parser.add_argument("--warmup_steps", type=int, default=10, help="number of warmup steps for timing stabilization")
    parser.add_argument("--steps", type=int, default=10, help="number of steps to profile")

    # model
    parser.add_argument("--model_size", type=str, default="small", choices=MODEL_CONFIGS.keys(),
                        help=f"model size configuration, one of: {list(MODEL_CONFIGS.keys())}")

    # system
    parser.add_argument("--seed", type=int, default=42, help="seed for the random number generator")
    parser.add_argument("--use_compile", action="store_true", help="use torch.compile to further speedup the model")
    parser.add_argument("--mixed_precision", action="store_true", help="use bfloat16 mixed precision")
    parser.add_argument("--use_tf32", action="store_true", help="use tf32 precision")

    # output
    parser.add_argument("--results_dir", type=str, default="results", help="directory to save benchmark results")
    parser.add_argument("--file_prefix", type=str, default="benchmark", help="prefix for output filenames")

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

    # model size info
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"Model {args.model_size}: {num_params/1e6:.2f}M, size: {model_size_mb:.2f} MB")

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

    fw_t = []
    bw_t = []
    opt_t = []

    model.train()
    print(f"Running {args.warmup_steps} warmup steps and {args.steps} measurement steps...")
    
    for step in range(max_steps):
        last_step = (step == max_steps - 1)
        optimizer.zero_grad(set_to_none=True)

        x, y = get_random_batch(args.batch_size, args.seq_len, vocab_size, DEVICE)

        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        
        # Forward Pass
        with ctx:
            _, loss = model(x, y)
        torch.cuda.synchronize()
        t1 = timeit.default_timer()

        # Backward Pass
        loss.backward()        
        torch.cuda.synchronize()
        t2 = timeit.default_timer()

        # Optimization step
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
        optimizer.step()
        torch.cuda.synchronize()
        t3 = timeit.default_timer()

        if step >= args.warmup_steps:
            fw_t.append((t1 - t0) * 1000)  # ms
            bw_t.append((t2 - t1) * 1000)  # ms
            opt_t.append((t3 - t2) * 1000) # ms

        if step % 5 == 0 or last_step:
            dt = (t3 - t0) * 1000
            toks_per_sec = (args.batch_size * args.seq_len * 1000) / dt
            print(f"step: {step:04d} | loss: {loss.item():.4f} | dt: {dt:.2f} ms | toks/sec: {toks_per_sec:.2f}")

    # Build per-step timing DataFrame
    total_t = [f + b + o for f, b, o in zip(fw_t, bw_t, opt_t)]
    toks_per_step = args.batch_size * args.seq_len

    df_steps = pd.DataFrame({
        "Forward (ms)": fw_t,
        "Backward (ms)": bw_t,
        "Optimizer (ms)": opt_t,
        "Total (ms)": total_t,
        "Tokens/s": [(toks_per_step * 1000) / t for t in total_t],
    })
    df_steps.index.name = "Step"

    # Build summary DataFrame
    fw_bw_t = [f + b for f, b in zip(fw_t, bw_t)]
    toks_per_sec = [(toks_per_step * 1000) / t for t in total_t]

    summary = pd.DataFrame({
        "Mean":  [np.mean(fw_t), np.mean(bw_t), np.mean(opt_t), np.mean(fw_bw_t), np.mean(total_t), np.mean(toks_per_sec)],
        "Std":   [np.std(fw_t),  np.std(bw_t),  np.std(opt_t),  np.std(fw_bw_t),  np.std(total_t),  np.std(toks_per_sec)],
        "Min":   [np.min(fw_t),  np.min(bw_t),  np.min(opt_t),  np.min(fw_bw_t),  np.min(total_t),  np.min(toks_per_sec)],
        "Max":   [np.max(fw_t),  np.max(bw_t),  np.max(opt_t),  np.max(fw_bw_t),  np.max(total_t),  np.max(toks_per_sec)],
    }, index=["Forward (ms)", "Backward (ms)", "Optimizer (ms)", "Fwd+Bwd (ms)", "Total (ms)", "Throughput (tok/s)"])

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({args.model_size} | {num_params/1e6:.2f}M params | {model_size_mb:.2f} MB)")
    print(f"{'='*60}")
    print(f"\nPer-Step Timings:")
    print(df_steps.to_string(float_format="%.2f"))
    print(f"\nSummary:")
    print(summary.to_string(float_format="%.2f"))
    print(f"{'='*60}\n")

    
    # Save to disk
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    config_yaml = os.path.join(results_dir, f"{args.file_prefix}_config_{timestamp}.yaml")
    steps_csv = os.path.join(results_dir, f"{args.file_prefix}_steps_{timestamp}.csv")
    summary_csv = os.path.join(results_dir, f"{args.file_prefix}_summary_{timestamp}.csv")
    
    # config
    config_metadata = build_config_metadata(args, gpt_config, num_params, model_size_mb)
    with open(config_yaml, 'w') as f:
        yaml.dump(config_metadata, f, default_flow_style=False, sort_keys=False)
    print(f"Saved configuration metadata to: {config_yaml}")
    
    # per-step datapoints
    df_steps.to_csv(steps_csv)
    print(f"Saved per-step benchmark data to: {steps_csv}")
    
    # summary statistics
    summary.to_csv(summary_csv)
    print(f"Saved benchmark summary to: {summary_csv}\n")


def main():
    """Main entry point."""
    args = parse_args()

    # print config
    for k, v in vars(args).items():
        print(f"{k:<20}: {v}")

    benchmark(args)


if __name__ == "__main__":
    main()