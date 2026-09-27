"""resume.pt: its format, its configuration, and the training device's CUDA generator."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import torch

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.paths import RunPaths
from mule_pattern_learner.training import trainer
from mule_pattern_learner.training.checkpoint import (
    ResumeState,
    load_resume_state,
    restore_cuda_rng,
    run_started,
)


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


def test_a_resume_state_loads_only_in_its_format_and_configuration(tmp_path: Path) -> None:
    run = RunPaths(tmp_path / "run")
    assert not run_started(run) and load_resume_state(DEFAULT_CONFIG, run) is None
    run.root.mkdir()
    ResumeState({"epoch": 1, "step": 0}).save(run.resume, DEFAULT_CONFIG)
    assert run_started(run)
    payload = torch.load(run.resume, weights_only=True)
    assert payload["format"] == ResumeState.FORMAT == 1
    assert payload["config_fingerprint"] == DEFAULT_CONFIG.fingerprint()
    state = load_resume_state(DEFAULT_CONFIG, run)
    assert state is not None and state.values == {"epoch": 1, "step": 0}
    # Runtime settings may change between segments; results settings may not.
    faster = DEFAULT_CONFIG.with_changes({"runtime": {"threads": 8}})
    assert load_resume_state(faster, run) == state
    slower = DEFAULT_CONFIG.with_changes({"training": {"learning_rate": 0.5}})
    with pytest.raises(ValueError, match="different configuration"):
        load_resume_state(slower, run)
    # A state of another layout, such as the checkpoint_last.pt of older runs, is refused.
    torch.save({**payload, "format": 0}, run.resume)
    with pytest.raises(ValueError, match="format 1"):
        load_resume_state(DEFAULT_CONFIG, run)
    run.metrics.write_text("{}")
    with pytest.raises(FileExistsError, match="already complete"):
        load_resume_state(DEFAULT_CONFIG, run)
