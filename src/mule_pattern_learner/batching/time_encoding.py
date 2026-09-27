"""The shared Fourier basis in torch, for edge ages the client computes on the device."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..contract.time_basis import FREQUENCY, SCALE

if TYPE_CHECKING:
    import torch


def fourier64_torch(delta_ms: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
    """`fourier64` on the tensor's device, returned as float32.

    Integer millisecond deltas are required. The arithmetic follows the numpy
    order of operations in float64 where the device supports it (CPU, CUDA) and
    in float32 on MPS; both agree with `fourier64` within 1e-5. `validate=False`
    skips the negativity check (a device sync) for deltas already checked on the host.
    """
    import torch

    if delta_ms.is_floating_point() or delta_ms.is_complex() or delta_ms.dtype == torch.bool:
        raise TypeError("Fourier deltas must be integer milliseconds")
    if validate and delta_ms.numel() and bool((delta_ms < 0).any()):
        raise ValueError("Future timestamps cannot be encoded as historical context")
    dtype = torch.float32 if delta_ms.device.type == "mps" else torch.float64
    frequency = torch.as_tensor(FREQUENCY, dtype=dtype, device=delta_ms.device)
    u = torch.log1p(delta_ms.to(dtype) / 1000.0) / SCALE
    angle = u[..., None] * frequency * (2 * np.pi)
    return (
        torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        .reshape(*delta_ms.shape, 64)
        .to(torch.float32)
    )
