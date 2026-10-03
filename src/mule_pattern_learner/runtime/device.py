"""The torch device, determinism and CPU threads of a command, restored when it ends."""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# cuBLAS needs a fixed workspace for deterministic GEMMs. CUDA reads it when the
# runtime initializes, so every entry point (the CLI and each script that loads torch)
# calls reserve_deterministic_cublas first; importing a module never sets it.
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


def attention_kernels(device: torch.device, deterministic: bool) -> AbstractContextManager[object]:
    """The attention kernels a run may use: only the math backend on a deterministic CUDA run.

    On CUDA, scaled dot-product attention (which nn.MultiheadAttention calls) picks a
    fused kernel, and the memory-efficient one has a backward pass that is not
    deterministic: with deterministic algorithms in warn-only mode torch warns and runs
    it anyway. The math backend computes the same attention from plain matrix products,
    whose backward is deterministic. Its cost is expected to be small here, though it is
    not yet timed on the CUDA host: each root is one query over 1 + fanout keys, so the
    attention matrix the math backend materialises is batch x heads x 1 x (1 + fanout),
    and the fused kernels save memory and time mainly on long sequences. CPU and MPS keep
    their kernels, so their numbers do not change.
    """
    if device.type == "cuda" and deterministic:
        return sdpa_kernel(SDPBackend.MATH)
    return nullcontext()


@contextmanager
def torch_runtime(
    device: torch.device, *, deterministic: bool | str = True, threads: int | None = None
) -> Generator[None, None, None]:
    """Apply determinism and CPU thread settings, then restore the global torch state.

    deterministic=True enables deterministic algorithms, warning instead of failing
    on CUDA-only gaps; "strict" fails everywhere; False leaves them off. On CUDA either
    deterministic mode also limits attention to its deterministic kernel
    (attention_kernels).
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
        with attention_kernels(device, enabled):
            yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.set_num_threads(previous[2])
