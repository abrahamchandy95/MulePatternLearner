"""The two files training saves: the selected model and the resume state.

model.pt is the selected model, as scoring and audits read it.
`_TrainingRun._model_payload` builds its payload. Readers load it once and pass
the ModelCheckpoint on; each checks only what it relies on (the configuration is
validated where TemporalPredictor, score_new_accounts and evaluate_final_population
use it, never on load).

run_dir/checkpoint_last.pt is the resume state that `_TrainingRun.save_last`
writes. A run resumes from it only when every setting that can change results
matches (resume_fingerprint); RUNTIME_KEYS may change between segments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import torch

from ..encoding import BASIS_ID
from .config_schema import validate_config
from .contract import FeaturePlan, contract_fingerprint, fingerprint
from .dataset import manifest_digest

CHECKPOINT_FORMAT = "temporal_live_checkpoint_v1"
# Transport, prefetch and logging settings never change results, so a resumed run may
# change them (for example a lower query concurrency after server trouble). The
# rejection limit only decides whether a run may go on, never its numbers, so it
# may be raised to resume a run that stopped on rejected roots.
RUNTIME_KEYS = frozenset(
    {
        "prefetch_batches",
        "checkpoint_every_steps",
        "log_every_steps",
        "request_batch_size",
        "query_concurrency",
        "context_lru_capacity",
        "encoding_check_every",
        "max_query_attempts",
        "max_outage_s",
        "max_rejected_root_fraction",
    }
)
# Files training writes into its run directory; any of them means the run has started.
RUN_STATE_FILES = ("config.json", "progress.jsonl", "checkpoint_last.pt", "metrics.json")


@dataclass(frozen=True)
class ModelCheckpoint:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> ModelCheckpoint:
        return cls(path, torch.load(path, map_location="cpu", weights_only=True))

    @classmethod
    def of(cls, checkpoint: Path | ModelCheckpoint) -> ModelCheckpoint:
        """An already loaded checkpoint, or the one at a path."""
        return checkpoint if isinstance(checkpoint, ModelCheckpoint) else cls.load(checkpoint)

    @property
    def config(self) -> dict[str, Any]:
        """The training configuration as saved; see validated_config."""
        return self.payload["config"]

    def validated_config(self) -> dict[str, Any]:
        return validate_config(self.config)

    @property
    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.payload["state_dict"]

    @property
    def threshold(self) -> float:
        return float(self.payload["threshold"])

    @property
    def selected_on(self) -> str | None:
        return self.payload.get("selected_on")

    @property
    def training_protocol(self) -> str | None:
        return self.payload.get("training_protocol")

    @property
    def dataset(self) -> Path | None:
        """The prepared dataset recorded at training time, if any."""
        value = self.payload.get("dataset")
        return Path(value) if value else None

    def check_contract(self) -> None:
        """Refuse a model saved under another feature or time-basis contract."""
        if (
            self.payload["contract"] != contract_fingerprint()
            or self.payload["basis_id"] != BASIS_ID
        ):
            raise ValueError("Checkpoint feature/time contract differs from this sampler")

    def check_inputs(self, plan: FeaturePlan) -> None:
        """Refuse a model whose input groups differ from those of its configuration."""
        if self.payload.get("input_fingerprint", plan.fingerprint()) != plan.fingerprint():
            raise ValueError("Checkpoint input groups differ from its configuration")

    def check_dataset(
        self,
        dataset: Path,
        message: str = "Checkpoint belongs to a different prepared dataset",
    ) -> None:
        """Refuse a prepared dataset other than the one this model was trained on."""
        if self.payload.get("dataset_manifest_sha256") != manifest_digest(dataset):
            raise ValueError(message)


def _result_view(config: dict[str, Any]) -> dict[str, Any]:
    """Settings that can change results, minus the sampler backend.

    The backend a run samples with is compared separately (the resolved backend
    is stored in the checkpoint), so a resume may name the new backend explicitly.
    """
    view = {k: v for k, v in config.items() if k not in RUNTIME_KEYS}
    sampler = view.pop("sampler", None)
    if isinstance(sampler, dict):
        sampler = {k: v for k, v in sampler.items() if k != "backend"}  # pyright: ignore[reportUnknownVariableType]
    # An empty [sampler] table means the defaults, like an absent one.
    if sampler:
        view["sampler"] = sampler
    return view


def resume_fingerprint(config: dict[str, Any]) -> str:
    """Fingerprint of every setting that can change a run's results."""
    return fingerprint(_result_view(config))


def explicit_backend(config: dict[str, Any]) -> str | None:
    """The sampler backend the config names, or None for auto and absent."""
    sampler = config.get("sampler") or {}
    backend = sampler.get("backend") if isinstance(sampler, dict) else None
    return None if backend in (None, "auto") else str(backend)  # pyright: ignore[reportUnknownArgumentType]


def restore_cuda_rng(saved: Any, device: torch.device) -> None:
    """Restore a saved CUDA generator state on any number of visible GPUs.

    Checkpoints hold the training device's state. A list (one state per device)
    restores only the devices that are visible now. Every training step reseeds
    all devices through torch.manual_seed, so nothing in training depends on it.
    """
    if saved is None or device.type != "cuda":
        return
    if isinstance(saved, torch.Tensor):
        torch.cuda.set_rng_state(saved, device)
        return
    for index, value in enumerate(list(saved)[: torch.cuda.device_count()]):  # pyright: ignore[reportUnknownArgumentType]
        torch.cuda.set_rng_state(value, index)  # pyright: ignore[reportUnknownArgumentType]


def atomic_save(value: dict[str, Any], path: Path) -> None:
    """Write a torch payload atomically, so a crash never leaves a truncated file."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_resume_state(config: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    """The saved state of an interrupted run, or None before its first checkpoint.

    A finished run and a configuration whose results-relevant settings changed
    are refused.
    """
    if (run_dir / "metrics.json").exists():
        raise FileExistsError(f"Run is already complete: {run_dir}")
    saved = run_dir / "config.json"
    if saved.exists():
        previous = _result_view(json.loads(saved.read_text()))
        current = _result_view(config)
        changed = sorted(
            key
            for key in set(previous) | set(current)
            if fingerprint(previous.get(key)) != fingerprint(current.get(key))
        )
        if changed:
            raise ValueError(f"Resumed configuration differs from the run: {changed}")
    path = run_dir / "checkpoint_last.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    if state["config_fingerprint"] != resume_fingerprint(config):
        raise ValueError("Checkpoint belongs to a different configuration")
    return state
