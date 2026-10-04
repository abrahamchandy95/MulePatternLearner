"""resume.pt: the resume state of an interrupted run.

`_TrainingRun.save_last` writes it as a ResumeState. A run resumes from it only when
every setting that can change results matches (config.RunConfig.fingerprint) and the
dataset is the one it trained on (check_run_dataset); the transport and runtime
sections, and the sampler backend, may change between segments (for example a lower
query concurrency after server trouble, or a higher rejection limit to resume a run
that stopped on rejected roots). A complete run is reported from its metrics.json
under the settings check (completed_run).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

import torch

from ..artifacts import atomic_write, read_json, read_run_config, read_run_provenance
from ..config import RunConfig, differing_settings
from ..paths import RunPaths


@dataclass(frozen=True)
class ResumeState:
    """What an interrupted run continues from, as resume.pt stores it.

    `_TrainingRun.save_last` writes it and `_TrainingRun.restore` continues from it.
    resume.pt stores its fields with FORMAT and the fingerprint of the configuration.
    """

    # The payload layout this code writes and reads.
    FORMAT: ClassVar[int] = 1

    model: dict[str, torch.Tensor]
    optimizer: dict[str, Any]
    # WeightAverage.saved(), or None without a weight average.
    weight_average: dict[str, Any] | None
    # The numpy generator at the start of the current epoch, whose schedule it draws.
    numpy_rng: Mapping[str, Any]
    torch_rng: torch.Tensor
    # The training device's CUDA generator, or None off CUDA.
    cuda_rng: torch.Tensor | None
    # The schedule position: epochs finished, steps of the current epoch done.
    epoch: int
    step: int
    stopped: bool
    loss_sum: torch.Tensor
    loss_steps: int
    # The selection so far: the kept epoch's weights, its value under the selection rule
    # (training.selection: higher is better, so the nnPU risk is negated; the field keeps
    # the name it had when the validation AP was the only rule), scores and mask.
    best_state: dict[str, torch.Tensor]
    best_ap: float
    best_epoch: int
    best_scores: torch.Tensor | None
    best_accepted: torch.Tensor | None
    # The epochs.csv rows of the finished epochs, without their selected flag; a state
    # saved before validation_pu_risk joined epochs.csv has rows without it.
    epoch_rows: list[dict[str, Any]]
    # The sampler backend the run resolved.
    sampler_backend: str
    # The dataset the run trains on: its id and its manifest's sha256.
    dataset_id: str
    dataset_manifest_sha256: str
    # history.RunTotals.saved(): the totals of every segment so far.
    progress: dict[str, Any]
    # inference.rejections.TrainingRejections.saved().
    rejections: dict[str, dict[str, int]]

    def save(self, path: Path, config: RunConfig) -> None:
        """Replace path atomically, so a crash never leaves a truncated state."""
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        payload |= {"format": self.FORMAT, "config_fingerprint": config.fingerprint()}
        with atomic_write(path) as pending:
            torch.save(payload, pending)

    @classmethod
    def load(cls, path: Path, config: RunConfig) -> ResumeState:
        """The state saved at path, refused unless config is the configuration it was saved by."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.pop("format", None) != cls.FORMAT:
            raise ValueError(f"{path} is not a resume state of format {cls.FORMAT}")
        if payload.pop("config_fingerprint", None) != config.fingerprint():
            raise ValueError("The resume state belongs to a different configuration")
        if set(payload) != {field.name for field in fields(cls)}:
            raise ValueError(f"{path} holds other fields than a resume state of this code")
        return cls(**payload)


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


def changed_settings(config: RunConfig, run: RunPaths) -> list[str]:
    """The dotted names of the results-relevant settings config changes from the run's."""
    try:
        previous = read_run_config(run.config)
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read the configuration of the run in {run.root}: {error}"
        ) from None
    return differing_settings(config.results_view(), previous.results_view())


def completed_run(config: RunConfig, run: RunPaths) -> dict[str, Any] | None:
    """The metrics.json record of a complete run of config, or None if the run is not complete.

    Nothing is written. A complete run whose results-relevant settings differ from
    config is an error that names the settings that changed.
    """
    if not run.metrics.exists():
        return None
    changed = changed_settings(config, run)
    if changed:
        raise ValueError(
            f"The run in {run.root} is complete with other settings: {changed}; "
            "move it aside to train these"
        )
    return read_json(run.metrics)


def check_resumable(config: RunConfig, run: RunPaths) -> None:
    """Refuse to resume the run with config if a results-relevant setting changed.

    The error names the settings that changed. A run that has not written its
    config.json yet has nothing to compare.
    """
    if run.config.exists():
        changed = changed_settings(config, run)
        if changed:
            raise ValueError(f"Resumed configuration differs from the run: {changed}")


def check_run_dataset(
    run: RunPaths, state: ResumeState | None, dataset_id: str, manifest_sha256: str
) -> None:
    """Refuse to continue a started run on a dataset other than the one it trained on.

    ``dataset_id`` and ``manifest_sha256`` name the dataset given now. The resume state
    names the run's dataset by both; a run stopped before its first checkpoint names
    its dataset id in config.json's provenance. A run that wrote neither has nothing
    to compare.
    """
    if state is not None:
        recorded = {"id": state.dataset_id, "manifest sha256": state.dataset_manifest_sha256}
    elif run.config.exists():
        recorded = {"id": read_run_provenance(run.config).get("dataset_id")}
    else:
        return
    current = {"id": dataset_id, "manifest sha256": manifest_sha256}
    changed = [
        f"{name} {recorded[name]} (given {current[name]})"
        for name in recorded
        if recorded[name] != current[name]
    ]
    if changed:
        raise ValueError(
            f"The run in {run.root} trained on another dataset: dataset {'; '.join(changed)}. "
            "Resume it on its own dataset, or train these settings into a new run"
        )


def load_resume_state(config: RunConfig, run: RunPaths) -> ResumeState | None:
    """The saved state of an interrupted run, or None before its first checkpoint.

    A configuration whose results-relevant settings changed is refused (check_resumable);
    a finished run is refused after that check.
    """
    check_resumable(config, run)
    if run.metrics.exists():
        raise FileExistsError(f"Run is already complete: {run.root}")
    if not run.resume.exists():
        return None
    return ResumeState.load(run.resume, config)
