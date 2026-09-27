"""resume.pt: the resume state of an interrupted run.

`_TrainingRun.save_last` writes it. A run resumes from it only when every setting
that can change results matches (config.RunConfig.fingerprint); the transport and
runtime sections, and the sampler backend, may change between segments (for example a
lower query concurrency after server trouble, or a higher rejection limit to resume a
run that stopped on rejected roots).
"""

from __future__ import annotations

import json
from typing import Any

import torch

from ..config import RunConfig, differing_settings
from ..paths import RunPaths

CHECKPOINT_FORMAT = "temporal_live_checkpoint_v1"


def run_started(run: RunPaths) -> bool:
    """Whether training wrote into the run's directory: any of its files means it has."""
    written = (run.config, run.events, run.resume, run.model, run.metrics)
    return any(path.exists() for path in written)


def restore_cuda_rng(saved: torch.Tensor | None, device: torch.device) -> None:
    """Restore the training device's saved CUDA generator state.

    Every training step reseeds all devices through torch.manual_seed, so nothing in
    training depends on it.
    """
    if saved is None or device.type != "cuda":
        return
    torch.cuda.set_rng_state(saved, device)


def load_resume_state(config: RunConfig, run: RunPaths) -> dict[str, Any] | None:
    """The saved state of an interrupted run, or None before its first checkpoint.

    A finished run and a configuration whose results-relevant settings changed
    are refused; the error names the settings that changed.
    """
    if run.metrics.exists():
        raise FileExistsError(f"Run is already complete: {run.root}")
    if run.config.exists():
        try:
            previous = RunConfig.from_dict(json.loads(run.config.read_text()))
        except ValueError as error:
            raise ValueError(
                f"Cannot read the configuration of the run in {run.root}: {error}"
            ) from None
        changed = differing_settings(config.results_view(), previous.results_view())
        if changed:
            raise ValueError(f"Resumed configuration differs from the run: {changed}")
    if not run.resume.exists():
        return None
    state = torch.load(run.resume, map_location="cpu", weights_only=True)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {run.resume}")
    if state["config_fingerprint"] != config.fingerprint():
        raise ValueError("Checkpoint belongs to a different configuration")
    return state
