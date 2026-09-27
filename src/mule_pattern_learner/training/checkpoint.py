"""run_dir/checkpoint_last.pt: the resume state of an interrupted run.

`_TrainingRun.save_last` writes it. A run resumes from it only when every setting
that can change results matches (config.RunConfig.fingerprint); the transport and
runtime sections, and the sampler backend, may change between segments (for example a
lower query concurrency after server trouble, or a higher rejection limit to resume a
run that stopped on rejected roots).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..config import RunConfig, differing_settings

CHECKPOINT_FORMAT = "temporal_live_checkpoint_v1"
# Files training writes into its run directory; any of them means the run has started.
RUN_STATE_FILES = ("config.json", "progress.jsonl", "checkpoint_last.pt", "metrics.json")


def restore_cuda_rng(saved: torch.Tensor | None, device: torch.device) -> None:
    """Restore the training device's saved CUDA generator state.

    Every training step reseeds all devices through torch.manual_seed, so nothing in
    training depends on it.
    """
    if saved is None or device.type != "cuda":
        return
    torch.cuda.set_rng_state(saved, device)


def load_resume_state(config: RunConfig, run_dir: Path) -> dict[str, Any] | None:
    """The saved state of an interrupted run, or None before its first checkpoint.

    A finished run and a configuration whose results-relevant settings changed
    are refused; the error names the settings that changed.
    """
    if (run_dir / "metrics.json").exists():
        raise FileExistsError(f"Run is already complete: {run_dir}")
    saved = run_dir / "config.json"
    if saved.exists():
        try:
            previous = RunConfig.from_dict(json.loads(saved.read_text()))
        except ValueError as error:
            raise ValueError(
                f"Cannot read the configuration of the run in {run_dir}: {error}"
            ) from None
        changed = differing_settings(config.results_view(), previous.results_view())
        if changed:
            raise ValueError(f"Resumed configuration differs from the run: {changed}")
    path = run_dir / "checkpoint_last.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    if state["config_fingerprint"] != config.fingerprint():
        raise ValueError("Checkpoint belongs to a different configuration")
    return state
