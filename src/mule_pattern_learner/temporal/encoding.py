"""The fixed Fourier basis shared with the installed GSQL encoder."""

import numpy as np
from numpy.typing import NDArray

BASIS_ID = "log1p_s_400d_32x_sincos_v1"


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
