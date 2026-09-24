from collections.abc import Generator
from contextlib import contextmanager
import os

import torch

# cuBLAS needs a fixed workspace for deterministic GEMMs. CUDA reads it when the
# runtime initializes, so entry points set it before any CUDA work starts.
CUBLAS_WORKSPACE = ":4096:8"


def reserve_deterministic_cublas() -> None:
    """Default CUBLAS_WORKSPACE_CONFIG for deterministic GEMMs; a user value wins."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)


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


@contextmanager
def torch_runtime(
    device: torch.device, *, deterministic: bool | str = True, threads: int | None = None
) -> Generator[None, None, None]:
    """Apply determinism and CPU thread settings, then restore the global torch state.

    deterministic=True enables deterministic algorithms, warning instead of failing
    on CUDA-only gaps; "strict" fails everywhere; False leaves them off.
    """
    if deterministic not in (True, False, "strict"):
        raise ValueError('deterministic must be true, false or "strict"')
    enabled = deterministic == "strict" or bool(deterministic)
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_num_threads(),
    )
    try:
        if device.type == "cuda" and enabled:
            reserve_deterministic_cublas()
        if threads is not None:
            torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(
            enabled, warn_only=deterministic != "strict" and device.type == "cuda"
        )
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.set_num_threads(previous[2])


# Keep the established public name for existing training callers.
select_device = choose_device
