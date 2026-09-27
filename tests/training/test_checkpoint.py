"""Restoring the training device's CUDA generator state."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch

from mule_pattern_learner.training import trainer
from mule_pattern_learner.training.checkpoint import restore_cuda_rng


def test_cuda_rng_restore_sets_the_training_device_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored: list[Any] = []

    def set_rng_state(state: torch.Tensor, device: int | torch.device = 0) -> None:
        restored.append(device)

    monkeypatch.setattr(torch.cuda, "set_rng_state", set_rng_state)
    state = torch.zeros(16, dtype=torch.uint8)
    restore_cuda_rng(state, torch.device("cuda", 0))
    assert restored == [torch.device("cuda", 0)]
    restore_cuda_rng(state, torch.device("cpu"))
    restore_cuda_rng(None, torch.device("cuda"))
    assert len(restored) == 1
    assert "get_rng_state_all" not in inspect.getsource(trainer)
