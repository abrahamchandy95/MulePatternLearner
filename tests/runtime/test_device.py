"""Device choice and the torch determinism settings."""

from __future__ import annotations

from collections.abc import Generator
import contextlib
import os
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention import SDPBackend

from mule_pattern_learner.runtime import device as runtime_device
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


def test_a_deterministic_cuda_run_attends_with_the_math_kernel_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    entered: list[SDPBackend] = []

    @contextlib.contextmanager
    def kernels(backends: SDPBackend) -> Generator[None]:
        entered.append(backends)
        yield

    monkeypatch.setattr(runtime_device, "sdpa_kernel", kernels)
    cuda = torch.device("cuda")  # a selection test only; CUDA is never initialized
    cases: list[tuple[torch.device, bool | str, list[SDPBackend]]] = [
        (cuda, True, [SDPBackend.MATH]),
        (cuda, "strict", [SDPBackend.MATH]),
        (cuda, False, []),
        # CPU and MPS keep their kernels, so the golden run's numbers stay as recorded.
        (torch.device("cpu"), True, []),
        (torch.device("mps"), True, []),
    ]
    for device, deterministic, expected in cases:
        entered.clear()
        with torch_runtime(device, deterministic=deterministic):
            assert entered == expected, (device, deterministic)


def test_the_math_kernel_holds_for_the_block_and_the_kernels_are_restored_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    backends = torch.backends.cuda
    flags = (
        backends.flash_sdp_enabled,
        backends.mem_efficient_sdp_enabled,
        backends.cudnn_sdp_enabled,
        backends.math_sdp_enabled,
    )
    before = [flag() for flag in flags]
    with torch_runtime(torch.device("cuda"), deterministic=True):
        assert [flag() for flag in flags] == [False, False, False, True]
    assert [flag() for flag in flags] == before
    with torch_runtime(torch.device("cpu"), deterministic=True):
        assert [flag() for flag in flags] == before


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
