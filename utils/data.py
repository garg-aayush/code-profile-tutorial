import torch
# -------------------------------------------------------------------------#
# data loader (random data)
# -------------------------------------------------------------------------#
def get_random_batch(B, T, vocab_size, device):
    """Generate a random batch of token IDs"""
    x, y = (torch.randint(0, vocab_size, (B, T)) for _ in range(2))
    if "cuda" in str(device):
        x,y = map(lambda k: k.pin_memory().to(device, non_blocking=True), [x,y])
    else:
        x,y = map(lambda k: k.to(device), [x,y])
    return x, y