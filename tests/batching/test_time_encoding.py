"""The torch Fourier basis matches the numpy one on every device."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mule_pattern_learner.batching.time_encoding import fourier64_torch
from mule_pattern_learner.contract.time_basis import fourier64

MPS = torch.backends.mps.is_available()


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if MPS else []))
def test_torch_fourier_matches_numpy(device: str) -> None:
    rng = np.random.default_rng(3)
    delta = np.concatenate(
        [[0, 1, 999, 86_400_000, 34_560_000_000, 2**40], rng.integers(0, 40_000_000_000, 5000)]
    )
    expected = fourier64(delta)
    actual = fourier64_torch(torch.from_numpy(delta).to(device)).cpu().numpy()
    assert actual.dtype == np.float32 and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    grid = fourier64_torch(torch.from_numpy(delta[:6].reshape(2, 3)).to(device))
    assert grid.shape == (2, 3, 64)
    with pytest.raises(ValueError, match="Future"):
        fourier64_torch(torch.tensor([-1], device=device))
    with pytest.raises(TypeError, match="integer"):
        fourier64_torch(torch.tensor([1.5], device=device))
