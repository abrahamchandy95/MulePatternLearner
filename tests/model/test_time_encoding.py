"""The Fourier basis: its constants, its values, the torch version against the numpy
reference on every device at hand, and the durations it refuses."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from mule_pattern_learner.model.time_encoding import (
    FREQUENCIES,
    SCALE,
    WIDTH,
    encode,
    encode_reference,
)

SECOND, DAY = 1_000, 86_400_000
TEN_YEARS = 3650 * DAY
# The CPU, and the accelerators this machine has.
DEVICES = [
    "cpu",
    *(["cuda"] if torch.cuda.is_available() else []),
    *(["mps"] if torch.backends.mps.is_available() else []),
]


def _durations() -> NDArray[np.int64]:
    """Hand-picked durations from none to a century, and random ones up to about 70 years."""
    picked = [0, 1, 999, SECOND, 3_600 * SECOND, DAY, 30 * DAY, 365 * DAY, TEN_YEARS]
    drawn = np.random.default_rng(0).integers(0, 2**41, 10_000)
    return np.concatenate([np.array(picked + [10 * TEN_YEARS], dtype=np.int64), drawn])


def test_the_basis_has_32_log_spaced_frequencies_from_an_eighth_to_two_cycles() -> None:
    assert len(FREQUENCIES) == 32 and WIDTH == 64
    assert (FREQUENCIES[0], FREQUENCIES[-1]) == pytest.approx((0.125, 2.0))
    ratios = FREQUENCIES[1:] / FREQUENCIES[:-1]
    assert np.allclose(ratios, 16 ** (1 / 31))
    assert SCALE == pytest.approx(math.log1p(10 * 365 * 86_400))


def test_a_duration_is_the_sines_then_the_cosines_of_its_scaled_log() -> None:
    """u = log1p(seconds) / log1p(10 years), and each angle is 2 pi f u."""
    for duration_ms in (0, 90 * SECOND, 40 * DAY, TEN_YEARS):
        u = math.log1p(duration_ms / 1000) / math.log1p(10 * 365 * 86_400)
        angle = 2 * math.pi * u * FREQUENCIES
        expected = np.concatenate([np.sin(angle), np.cos(angle)])
        assert np.allclose(encode_reference(np.array(duration_ms)), expected, atol=1e-12)
    # No time at all: every sine 0 and every cosine 1. Ten years: u is 1.
    assert np.array_equal(encode_reference(np.array(0)), [0.0] * 32 + [1.0] * 32)
    ten_years = encode_reference(np.array(TEN_YEARS))
    assert np.allclose(ten_years[:32], np.sin(2 * np.pi * FREQUENCIES))


@pytest.mark.parametrize("device", DEVICES)
def test_the_torch_encoding_agrees_with_the_reference(device: str) -> None:
    durations = _durations()
    expected = encode_reference(durations)
    found = encode(torch.as_tensor(durations, device=device))
    assert found.device.type == device and found.dtype == torch.float32
    assert found.shape == (len(durations), WIDTH)
    assert np.abs(found.cpu().numpy() - expected).max() < 1e-5


def test_the_encoding_keeps_the_shape_of_its_durations() -> None:
    durations = _durations()[:24].reshape(2, 3, 4)
    assert encode_reference(durations).shape == (2, 3, 4, WIDTH)
    assert encode(torch.as_tensor(durations)).shape == (2, 3, 4, WIDTH)
    assert encode(torch.zeros(0, 3, dtype=torch.long)).shape == (0, 3, WIDTH)
    found = encode(torch.as_tensor(durations)).numpy()
    assert np.abs(found - encode_reference(durations)).max() < 1e-5


def test_a_negative_duration_is_an_error() -> None:
    with pytest.raises(ValueError, match="negative"):
        encode_reference(np.array([DAY, -1]))
    with pytest.raises(ValueError, match="negative"):
        encode(torch.tensor([DAY, -1]))


def test_a_duration_must_be_integer_milliseconds() -> None:
    with pytest.raises(TypeError, match="integer milliseconds"):
        encode_reference(np.array([1.5]))
    with pytest.raises(TypeError, match="integer milliseconds"):
        encode_reference(np.array([True]))
    for wrong in (torch.tensor([1.5]), torch.tensor([True])):
        with pytest.raises(TypeError, match="integer milliseconds"):
            encode(wrong)
