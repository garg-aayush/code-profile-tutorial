#!/bin/bash

echo "E2E Benchmark"

# Run with warmup
uv run benchmark_gpt2.py --model medium --warmup_steps 5 --file_prefix e2e_warmup5 --results results/e2e
# Run with warmup
uv run benchmark_gpt2.py --model medium --warmup_steps 2 --file_prefix e2e_warmup2 --results results/e2e
# Run without warmup
uv run benchmark_gpt2.py --model medium --warmup_steps 0 --file_prefix e2e_warmup0 --results results/e2e
