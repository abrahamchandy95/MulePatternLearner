import torch


def choose_device(preferred: str | torch.device | None = None) -> torch.device:
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU.

    Returns a torch.device the caller moves the model and every batch onto. The
    order reflects throughput for batched tensor math; CPU is the portable
    fallback. torch.backends.mps.is_available() returns False on non-macOS
    builds, so the MPS branch is simply skipped there.
    """
    if preferred is not None and str(preferred) != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable to this process")
        if device.type not in {"cuda", "mps", "cpu"}:
            raise ValueError(f"Unsupported training device: {device}")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Keep the established public name for existing training callers.
select_device = choose_device
