"""The numpy Fourier basis matches the GSQL encoder and refuses future times."""

from __future__ import annotations

import numpy as np
import pytest

from mule_pattern_learner.contract.time_basis import fourier64


def test_fourier_matches_gsql_basis_and_rejects_negative() -> None:
    actual = fourier64(np.array([0, 1000, 34_560_000_000], dtype=np.int64))
    np.testing.assert_array_equal(actual[0, ::2], 0)
    np.testing.assert_array_equal(actual[0, 1::2], 1)
    frequencies = 0.125 * 16 ** (np.arange(32) / 31)
    np.testing.assert_allclose(actual[2, ::2], np.sin(2 * np.pi * frequencies), atol=1e-6)
    np.testing.assert_allclose(actual[2, 1::2], np.cos(2 * np.pi * frequencies), atol=1e-6)
    with pytest.raises(ValueError):
        fourier64(np.array([-1], dtype=np.int64))
