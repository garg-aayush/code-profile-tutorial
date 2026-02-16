# -------------------------------------------------------------#
# constants
# -------------------------------------------------------------#
MAX_LR = 1.5e-3           # constant learning rate
WEIGHT_DECAY = 0.1        # weight decay for optimizer
BETA1 = 0.9               # beta1 for optimizer
BETA2 = 0.95              # beta2 for optimizer
GRAD_NORM_CLIP = 1.0      # global norm gradient clipping
DEVICE = "cuda"           # device to use

# -------------------------------------------------------------#
# model configs
# -------------------------------------------------------------#
MODEL_CONFIGS = {
    "small":  dict(n_embd=768,  n_head=12, n_layer=12, vocab_size=10_000),
    "medium": dict(n_embd=1024, n_head=16, n_layer=24, vocab_size=10_000),
    "large":  dict(n_embd=1280, n_head=20, n_layer=36, vocab_size=10_000),
    "xl":     dict(n_embd=1600, n_head=25, n_layer=48, vocab_size=10_000),
    "xxl":    dict(n_embd=2560, n_head=32, n_layer=32, vocab_size=10_000),
}