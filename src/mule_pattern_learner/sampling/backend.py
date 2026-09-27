"""Per-step neighbor resampling from bounded, cutoff-safe candidate pools.

TigerGraph returns a bounded candidate pool per context (`PoolPlan`). The client
draws, per context and relation, a uniform subset without replacement and merges it
into the fanout slots, reserving a few for associations. Two interchangeable subset
samplers exist:

- `TorchGroupedSampler`: random keys plus a segmented rank; runs on CPU, MPS or CUDA.
- `CuGraphSampler`: pylibcugraph's heterogeneous temporal sampler on one CUDA GPU.

Evaluation always uses device-independent hash keys on the torch path, so scores do
not depend on the machine or the backend. The hop enters those keys through
`hop_seed(evaluation_seed, hop)`, so a root's hop-2 draw is independent of its hop-1
draw, as in training.

Training draws depend on the backend. The torch sampler takes its keys from a CPU
`torch.Generator` seeded by the step seed, so with `backend = "torch"` a fixed seed
reproduces the same neighborhoods on every device. The cuGraph sampler draws with
cuGraph's own RNG (`random_state` derived from the same step seed): its subsets are
equally distributed (uniform without replacement per context and relation) but
differ from the torch sampler's for the same seed. Compare runs across machines
with `backend = "torch"`, and resolve the backend once per run (`resolve_backend`).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from ..contract.sampler_plan import SamplerPlan
from .candidates import CandidateTable, hop_seed, merge_slots, relation_quotas, selection_keys
from .cugraph_sampler import (
    PYLIBCUGRAPH_PIN,
    CuGraphSampler,
    cugraph_usable,
    default_cugraph_sampler,
)
from .torch_sampler import TorchGroupedSampler


def select_resampled(
    candidates: CandidateTable,
    *,
    hop: int,
    sampler: SamplerPlan,
    fanout: int,
    mode: str = "eval",
    step_seed: int = 0,
    backend: str = "torch",
    device: str | torch.device = "cpu",
    cugraph: CuGraphSampler | None = None,
) -> np.ndarray:
    """[C, fanout] candidate rows per context, prefix-filled, -1 padded.

    `backend` must already be resolved (`resolve_backend`). Evaluation always uses
    the hash-keyed torch path; `device` is where the torch sampler runs. In training,
    `cugraph` and `torch` give different (equally distributed) subsets for one
    `step_seed`; the slot order always comes from the torch keys.
    """
    if hop not in (1, 2) or fanout < 1:
        raise ValueError("Hop must be 1 or 2 and fanout positive")
    if mode not in ("train", "eval"):
        raise ValueError("Sampler mode must be train or eval")
    if backend not in ("torch", "cugraph"):
        raise ValueError("Resolve the sampler backend before selecting")
    if not len(candidates):
        return np.full((candidates.num_contexts, fanout), -1, dtype=np.int64)
    quotas = relation_quotas(sampler, hop)
    keys = selection_keys(
        candidates,
        mode=mode,
        step_seed=step_seed,
        evaluation_seed=sampler.evaluation_seed,
        hop=hop,
    )
    if mode == "train" and backend == "cugraph":
        engine = cugraph or default_cugraph_sampler()
        keep = torch.from_numpy(
            engine.subset(candidates, quotas, random_state=hop_seed(step_seed, hop), device=device)
        )
    else:
        keep = TorchGroupedSampler(device).subset(candidates, quotas, keys)
    return merge_slots(
        candidates,
        keep,
        keys.to(keep.device),
        hop=hop,
        fanout=fanout,
        association_slots=sampler.association_slots,
    )


def resolve_backend(sampler: SamplerPlan, device: str | torch.device) -> str:
    """The subset backend of a run: "torch" or "cugraph".

    `torch` is always torch.
    `auto` is cugraph only on a CUDA device whose cached functional probe
    (`cugraph_usable`) passed; otherwise torch, with a warning when cuGraph is
    installed but failed the probe. Explicit `cugraph` raises with the probe's
    reason. The backends draw different (equally distributed) subsets for one step
    seed, so resolve once per run on the main thread and pass the result to every
    `make_live_batch(sampler_backend=...)` call.
    """
    if sampler.backend == "torch":
        return "torch"
    device = torch.device(device)
    if device.type != "cuda":
        if sampler.backend == "cugraph":
            raise RuntimeError(
                f"Sampler backend cugraph needs a CUDA device, got {device}; "
                'set [sampler] backend = "torch" or "auto" for this host'
            )
        return "torch"
    # torch stubs type the index as int, but an unindexed CUDA device has None.
    index = device.index
    if index is None:  # pyright: ignore[reportUnnecessaryComparison]
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    probe = cugraph_usable(index)
    if probe.usable:
        return "cugraph"
    if sampler.backend == "cugraph":
        raise RuntimeError(
            f"Sampler backend cugraph cannot run on cuda:{index}: {probe.reason}. "
            f"Install {PYLIBCUGRAPH_PIN} and cupy, run "
            "scripts/temporal/verify_cugraph_sampler.py on the GPU host, "
            'or set [sampler] backend = "torch"'
        )
    if probe.installed:
        warnings.warn(
            f"cuGraph probe failed on cuda:{index} ({probe.reason}); using the torch "
            'sampler. Set [sampler] backend = "torch" to silence this, or "cugraph" '
            "to require cuGraph.",
            RuntimeWarning,
            stacklevel=2,
        )
    return "torch"


def batch_backend(
    sampler: SamplerPlan, device: torch.device, mode: str, resolved: str | None = None
) -> str:
    """The backend one batch selects with: "torch" or "cugraph".

    `resolved` is the run's `resolve_backend(sampler, device)` result; None resolves
    here (the probe is cached). Evaluation always runs the hash-keyed torch path, so
    there a resolved `cugraph` is accepted on any device and not used.
    """
    if resolved is not None:
        allowed = ("torch", "cugraph") if sampler.backend == "auto" else (sampler.backend,)
        if resolved not in allowed:
            raise ValueError(
                f"sampler_backend {resolved!r} does not fit the sampler backend "
                f"{sampler.backend!r}; expected one of {allowed}"
            )
    if mode == "eval":
        return "torch"
    if resolved is None:
        return resolve_backend(sampler, device)
    if resolved == "cugraph" and device.type != "cuda":
        raise ValueError(f"sampler_backend cugraph needs a CUDA batch device, got {device}")
    return resolved
