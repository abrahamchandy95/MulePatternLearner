"""The fixed Fourier basis every duration reaches the model in (docs/design.md, "Time").

A duration of s seconds has u = log1p(s) / log1p(10 years), so seconds, days and years all
spread over about [0, 1]. Its encoding is the sine and the cosine of 2 pi f u for 32
frequencies f, log-spaced from 0.125 to 2 cycles per unit of u: 64 numbers, the 32 sines
then the 32 cosines. A duration is integer milliseconds and never negative, since a
negative one would reach past the as-of time; either mistake is an error.

encode_reference is the reference, in numpy and float64. encode computes the same on its
tensor's device (the GPU in training) in float32, and agrees with it to within 1e-5.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

# Ten years of 365 days, in seconds: a duration of ten years has u = 1.
TEN_YEARS_S = 10 * 365 * 86_400
SCALE = math.log1p(TEN_YEARS_S)
# Cycles per unit of u.
FREQUENCIES = np.geomspace(0.125, 2.0, 32)
# The numbers that encode one duration.
WIDTH = 2 * len(FREQUENCIES)


def encode_reference(duration_ms: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
    """The encoding of each duration of an integer array, in a new last axis of WIDTH."""
    durations = np.asarray(duration_ms)
    if not np.issubdtype(durations.dtype, np.integer):
        raise TypeError(f"Durations are integer milliseconds, not {durations.dtype}")
    if np.any(durations < 0):
        raise ValueError("A duration is negative")
    u = np.log1p(durations.astype(np.float64) / 1000.0) / SCALE
    angle = 2 * np.pi * u[..., None] * FREQUENCIES
    return np.concatenate([np.sin(angle), np.cos(angle)], axis=-1)


def encode(duration_ms: Tensor) -> Tensor:
    """encode_reference of an integer tensor, in float32 on the tensor's device."""
    if (
        duration_ms.is_floating_point()
        or duration_ms.is_complex()
        or duration_ms.dtype == torch.bool
    ):
        raise TypeError(f"Durations are integer milliseconds, not {duration_ms.dtype}")
    if bool((duration_ms < 0).any()):
        raise ValueError("A duration is negative")
    frequencies = torch.as_tensor(FREQUENCIES, dtype=torch.float32, device=duration_ms.device)
    u = torch.log1p(duration_ms.to(torch.float32) / 1000.0) / SCALE
    angle = 2 * math.pi * u[..., None] * frequencies
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
