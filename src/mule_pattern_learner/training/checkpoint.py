"""run_dir/checkpoint_last.pt: the resume state of an interrupted run.

`_TrainingRun.save_last` writes it. A run resumes from it only when every setting
that can change results matches (resume_fingerprint); RUNTIME_KEYS may change
between segments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..contract.fingerprints import fingerprint

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


def restore_cuda_rng(saved: torch.Tensor | None, device: torch.device) -> None:
    """Restore the training device's saved CUDA generator state.

    Every training step reseeds all devices through torch.manual_seed, so nothing in
    training depends on it.
    """
    if saved is None or device.type != "cuda":
        return
    torch.cuda.set_rng_state(saved, device)


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
