"""resume.pt: the resume state of an interrupted run.

`_TrainingRun.save_last` writes it as a ResumeState. A run resumes from it only when
every setting that can change results matches (config.RunConfig.fingerprint); the
transport and runtime sections, and the sampler backend, may change between segments
(for example a lower query concurrency after server trouble, or a higher rejection
limit to resume a run that stopped on rejected roots).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from ..artifacts import atomic_write, read_run_config
from ..config import RunConfig, differing_settings
from ..paths import RunPaths


@dataclass(frozen=True)
class ResumeState:
    """What an interrupted run continues from.

    ``values`` holds the model, optimizer, weight average, RNG states, schedule
    position and selection state, and the run's totals; `_TrainingRun.save_last` names
    them. resume.pt stores them with FORMAT and the fingerprint of the configuration.
    """

    # The payload layout this code writes and reads.
    FORMAT: ClassVar[int] = 1

    values: dict[str, Any]

    def save(self, path: Path, config: RunConfig) -> None:
        """Replace path atomically, so a crash never leaves a truncated state."""
        payload = {**self.values, "format": self.FORMAT, "config_fingerprint": config.fingerprint()}
        with atomic_write(path) as pending:
            torch.save(payload, pending)

    @classmethod
    def load(cls, path: Path, config: RunConfig) -> ResumeState:
        """The state saved at path, refused unless config is the configuration it was saved by."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != cls.FORMAT:
            raise ValueError(f"{path} is not a resume state of format {cls.FORMAT}")
        if payload.pop("config_fingerprint") != config.fingerprint():
            raise ValueError("The resume state belongs to a different configuration")
        del payload["format"]
        return cls(payload)


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


def load_resume_state(config: RunConfig, run: RunPaths) -> ResumeState | None:
    """The saved state of an interrupted run, or None before its first checkpoint.

    A finished run and a configuration whose results-relevant settings changed
    are refused; the error names the settings that changed.
    """
    if run.metrics.exists():
        raise FileExistsError(f"Run is already complete: {run.root}")
    if run.config.exists():
        try:
            previous = read_run_config(run.config)
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Cannot read the configuration of the run in {run.root}: {error}"
            ) from None
        changed = differing_settings(config.results_view(), previous.results_view())
        if changed:
            raise ValueError(f"Resumed configuration differs from the run: {changed}")
    if not run.resume.exists():
        return None
    return ResumeState.load(run.resume, config)
