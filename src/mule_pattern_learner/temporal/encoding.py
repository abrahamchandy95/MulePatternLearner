"""The fixed Fourier basis shared with the installed GSQL encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch

BASIS_ID = "log1p_s_400d_32x_sincos_v1"
# log1p(400 days in seconds): deltas up to 400 days map to u in [0, 1].
_SCALE = float(np.log1p(34_560_000.0))
_FREQUENCY = 0.125 * np.power(16.0, np.arange(32, dtype=np.float64) / 31.0)


def fourier64(delta_ms: NDArray[np.int64]) -> NDArray[np.float32]:
    delta = np.asarray(delta_ms, dtype=np.int64)
    if np.any(delta < 0):
        raise ValueError("Future timestamps cannot be encoded as historical context")
    u = np.log1p(delta.astype(np.float64) / 1000.0) / np.log1p(34_560_000.0)
    frequency = 0.125 * np.power(16.0, np.arange(32, dtype=np.float64) / 31.0)
    angle = u[..., None] * frequency * (2 * np.pi)
    return (
        np.stack((np.sin(angle), np.cos(angle)), axis=-1)
        .reshape(*delta.shape, 64)
        .astype(np.float32)
    )


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
    frequency = torch.as_tensor(_FREQUENCY, dtype=dtype, device=delta_ms.device)
    u = torch.log1p(delta_ms.to(dtype) / 1000.0) / _SCALE
    angle = u[..., None] * frequency * (2 * np.pi)
    return (
        torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        .reshape(*delta_ms.shape, 64)
        .to(torch.float32)
    )
