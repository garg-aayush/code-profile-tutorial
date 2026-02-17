import torch
from utils.constants import DEVICE


def build_config_metadata(args, gpt_config, num_params, model_size_mb):
    """Build a configuration metadata dictionary for benchmark results."""
    return {
        "model": {
            "size": args.model_size,
            "parameters": int(num_params),
            "parameters_millions": round(num_params / 1e6, 2),
            "size_mb": round(model_size_mb, 2),
            "config": {
                "n_layer": gpt_config.n_layer,
                "n_embd": gpt_config.n_embd,
                "n_head": gpt_config.n_head,
                "vocab_size": gpt_config.vocab_size,
                "block_size": gpt_config.block_size,
            }
        },
        "training": {
            "batch_size": args.batch_size,
            "sequence_length": args.seq_len,
            "warmup_steps": args.warmup_steps,
            "measurement_steps": args.steps,
            "seed": args.seed,
        },
        "optimization": {
            "mixed_precision": args.mixed_precision,
            "use_compile": args.use_compile,
            "use_tf32": args.use_tf32,
        },
        "device": {
            "type": DEVICE,
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        }
    }
