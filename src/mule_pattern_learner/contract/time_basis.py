"""The fixed Fourier basis shared with the installed GSQL encoder."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BASIS_ID = "log1p_s_400d_32x_sincos_v1"
# log1p(400 days in seconds): deltas up to 400 days map to u in [0, 1].
SCALE = float(np.log1p(34_560_000.0))
FREQUENCY = 0.125 * np.power(16.0, np.arange(32, dtype=np.float64) / 31.0)


def fourier64(delta_ms: NDArray[np.int64]) -> NDArray[np.float32]:
    delta = np.asarray(delta_ms, dtype=np.int64)
    if np.any(delta < 0):
        raise ValueError("Future timestamps cannot be encoded as historical context")
    u = np.log1p(delta.astype(np.float64) / 1000.0) / SCALE
    angle = u[..., None] * FREQUENCY * (2 * np.pi)
    return (
        np.stack((np.sin(angle), np.cos(angle)), axis=-1)
        .reshape(*delta.shape, 64)
        .astype(np.float32)
    )
