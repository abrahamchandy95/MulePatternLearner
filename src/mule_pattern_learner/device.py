import torch


def select_device() -> torch.device:
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU.

    Returns a torch.device the caller moves the model and every batch onto. The
    order reflects throughput for batched tensor math; CPU is the portable
    fallback. torch.backends.mps.is_available() returns False on non-macOS
    builds, so the MPS branch is simply skipped there.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
