"""Device choice and the torch determinism settings."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from mule_pattern_learner.runtime.device import choose_device, torch_runtime


def test_torch_runtime_applies_modes_and_restores_global_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_num_threads(),
    )
    cuda = torch.device("cuda")  # a flag test only; CUDA is never initialized
    with torch_runtime(cuda, deterministic=True, threads=1):
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_num_threads() == 1
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    with torch_runtime(cuda, deterministic="strict"):
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    with torch_runtime(torch.device("cpu"), deterministic=True):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    with pytest.raises(RuntimeError), torch_runtime(torch.device("cpu"), deterministic=False):
        assert not torch.are_deterministic_algorithms_enabled()
        raise RuntimeError
    after = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_num_threads(),
    )
    assert after == before
    with pytest.raises(ValueError):
        with torch_runtime(torch.device("cpu"), deterministic="sometimes"):
            pass


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_choose_device_prefers_available_accelerator(cuda: bool, mps: bool, expected: str) -> None:
    with (
        patch("torch.cuda.is_available", return_value=cuda),
        patch("torch.backends.mps.is_available", return_value=mps),
    ):
        assert choose_device().type == expected
        assert choose_device("auto").type == expected
        assert choose_device("cpu").type == "cpu"
        if not mps:
            with pytest.raises(ValueError, match="MPS"):
                choose_device("mps")
