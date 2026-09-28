"""Read-only readiness of the graph and the built-in run, which `mule check` prints.

check connects with the run's transport section (the connection refuses a graph other
than contract.server.GRAPH_NAME) and reports whether the scope vertex type exists, which training
queries are installed with the repository text and, on a CUDA host, what the cuGraph
probe found. When all of them are ready and the run's dataset is prepared in data/, it
builds the first training batch the way train() builds it and runs one optimizer step
(first_step). Its source has no disk tier: the batch's contexts are requested from
TigerGraph, not read from the dataset's context cache, so the installed context query
and its first Fourier spot check run, and the REST calls and seconds are the graph's.
The report has the digest of every batch tensor
(batching.assemble.tensor_digests, the definition the golden-run test pins) and the
step's loss and objective, the first values train() logs. Two code versions built the
same batch and step when both print the same digests and loss on one machine and
device. Nothing is written to the graph and no dataset is prepared: `mule train` does
that.
"""

from __future__ import annotations

from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch

from ..batching.assemble import RootBatch, batch_device, build_root_batch, tensor_digests, to_device
from ..config import DEFAULT_CONFIG, RunConfig
from ..contract.server import TRAINING_QUERY_FILES
from ..data.contexts import ContextReader, ContextSource, close_source
from ..data.hub_registry import load_hub_registry
from ..data.manifest import dataset_mismatches, load_prepared
from ..data.observed_labels import load_observed_labels
from ..data.splits import sample_keys
from ..model.build import build_model
from ..model.loss import NonNegativePULoss
from ..paths import DATA_DIR, DatasetPaths
from ..runtime.device import choose_device, torch_runtime
from ..sampling.cugraph_sampler import cugraph_usable
from ..tigergraph.context_query import TigerGraphContextFetcher
from ..tigergraph.executor import ConnectionExecutor, TigerGraphExecutor
from ..tigergraph.gsql_text import repository_queries
from ..tigergraph.installer import has_scope_vertex, query_problems
from ..training.objective import nnpu_objective, nnpu_step
from ..training.schedule import TrainingStep, epoch_schedule
from ..training.trainer import build_optimizer, check_limits, training_samples
from .connect import connect, open_context_source
from .prepare import find_datasets


def rest_calls(contexts: ContextReader) -> tuple[int, dict[str, int]]:
    """Successful REST calls and retries of the graph's executor (0 without one, e.g. a fake)."""
    fetcher = contexts.fetcher if isinstance(contexts, ContextSource) else None
    executor = fetcher.executor if isinstance(fetcher, TigerGraphContextFetcher) else None
    if not isinstance(executor, TigerGraphExecutor):
        return 0, {}
    return executor.calls, dict(executor.retries)


def graph_readiness(executor: ConnectionExecutor) -> dict[str, Any]:
    """The graph name, the scope vertex type and the installed training queries."""
    problems = query_problems(executor)
    names = repository_queries(TRAINING_QUERY_FILES)
    return {
        "graph": executor.graph_name,
        "scope_schema": "present" if has_scope_vertex(executor) else "missing",
        "queries": {
            "up_to_date": [name for name in names if name not in problems],
            "stale": problems,
        },
    }


def cugraph_readiness() -> dict[str, Any]:
    """The cuGraph probe that `sampler.backend = "auto"` runs, on a CUDA host only."""
    if not torch.cuda.is_available():
        return {"status": "no_cuda"}
    index = torch.cuda.current_device()
    probe = cugraph_usable(index)
    status = "passed" if probe.usable else "installed_but_failed" if probe.installed else "missing"
    return {"status": status, "device": f"cuda:{index}", "reason": probe.reason or None}


def first_step(config: RunConfig, dataset: DatasetPaths, contexts: ContextReader) -> dict[str, Any]:
    """Build the first scheduled training batch of a dataset and run one optimizer step.

    The batch is epoch 1, step 1 of train()'s schedule: the run's sampler in training
    mode with the step seed, hub stubs from the dataset's registry, and rejected roots
    dropped. The step uses train()'s model, loss and seeding on the configured device
    under the run's determinism.
    """
    manifest, accounts = load_prepared(dataset)
    changed = dataset_mismatches(config, manifest)
    if changed:
        raise ValueError(f"Configuration differs from the prepared dataset: {changed}")
    plan = config.feature_plan()
    sampler, training, runtime = config.sampler, config.training, config.runtime
    check_limits(config, plan)
    device = choose_device(runtime.device)
    hubs = load_hub_registry(dataset, manifest)
    samples = training_samples(
        config.dataset.dates, accounts, load_observed_labels(accounts, dataset)
    )
    if not samples:
        raise ValueError("No train cutoff has revealed positives; train() would refuse too")
    step = epoch_schedule(
        samples,
        np.random.default_rng(training.seed),
        training.batch_size,
        epoch=0,
        seed=training.seed,
        max_steps=1,
    )[0]
    keys = sample_keys(accounts.iloc[step.indices], step.date, manifest)
    with torch_runtime(device, deterministic=runtime.deterministic, threads=runtime.threads):
        calls_before, _ = rest_calls(contexts)
        requests_before = contexts.database_calls
        started = time.perf_counter()
        prepared = build_root_batch(
            contexts,
            keys,
            fanouts=sampler.fanouts,
            device=batch_device(device),
            plan=plan,
            sampler=sampler,
            hubs=hubs,
            mode="train",
            step_seed=step.seed,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        build_seconds = time.perf_counter() - started
        calls_after, retries = rest_calls(contexts)
        report: dict[str, Any] = {
            "device": str(device),
            "deterministic": runtime.deterministic,
            "step_seed": step.seed,
            "roots": len(keys),
            "accepted_roots": int(prepared.accepted.sum()),
            "batch": prepared.stats,
            "rejections": dict(contexts.rejections),
            "context_requests": contexts.database_calls - requests_before,
            "rest_calls": calls_after - calls_before,
            "retries": retries,
            "batch_seconds": build_seconds,
            "input_fingerprint": plan.fingerprint(),
            "sampler_fingerprint": sampler.fingerprint(),
        }
        if prepared.batch is None:
            return {**report, "status": "every_root_rejected"}
        batch = prepared.batch
        report["status"] = "passed"
        report["tensor_bytes"] = sum(t.numel() * t.element_size() for t in batch.values())
        report["tensor_digests"] = tensor_digests(batch)
        report |= train_step(config, device, batch, prepared, step)
    if device.type == "mps":
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        report["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return report


def train_step(
    config: RunConfig,
    device: torch.device,
    batch: dict[str, torch.Tensor],
    prepared: RootBatch,
    step: TrainingStep,
) -> dict[str, Any]:
    """One nnPU optimizer step with the trainer's model, loss and seeding."""
    torch.manual_seed(config.training.seed)
    plan = config.feature_plan()
    model = build_model(config.model, plan, config.sampler.fanouts[0]).to(device)
    optimizer = build_optimizer(model, config.training)
    prior, positive_weight = nnpu_objective(config.loss)
    loss_fn = NonNegativePULoss(prior=prior, positive_weight=positive_weight)
    started = time.perf_counter()
    # Rejected roots were dropped; the leading accepted rows are the positives.
    positives = int(prepared.accepted[: len(step.positives)].sum())
    loss = nnpu_step(model, optimizer, loss_fn, to_device(batch, device), positives, step.seed)
    value = float(loss.value.cpu())  # waits for the accelerator
    if not np.isfinite(value):
        raise ValueError("Non-finite training loss in the first step")
    return {
        "loss": value,
        # The unclamped risk; it differs from the loss when the nnPU correction fired.
        "objective": float(loss.objective.cpu()),
        "train_step_seconds": time.perf_counter() - started,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def check(config: RunConfig = DEFAULT_CONFIG, data: Path = DATA_DIR) -> dict[str, Any]:
    """The readiness report of the built-in run; status "ready" once every check passed.

    The first step runs only on a graph that is ready, with the run's one dataset in
    data; otherwise the report names what is missing.
    """
    executor = connect(config.transport)
    report: dict[str, Any] = {**graph_readiness(executor), "cugraph": cugraph_readiness()}
    problems: list[str] = []
    if report["scope_schema"] == "missing":
        problems.append("the scope vertex type is missing; `mule install` adds it")
    if report["queries"]["stale"]:
        problems.append("training queries differ from the repository; `mule install` installs them")
    if config.sampler.backend == "cugraph" and report["cugraph"]["status"] != "passed":
        problems.append(f"sampler.backend is cugraph, but the probe found {report['cugraph']}")
    found = find_datasets(config, data)
    report["dataset"] = found[0].root.name if len(found) == 1 else None
    if len(found) != 1:
        problems.append(
            f"found {len(found)} datasets of the built-in run in {data}; `mule train` prepares one"
        )
    if not problems:
        (dataset,) = found
        manifest, _ = load_prepared(dataset)
        started = time.perf_counter()
        contexts = open_context_source(dataset, manifest, config, cached=False)
        report["source_open_seconds"] = time.perf_counter() - started
        failed = True
        try:
            report["first_step"] = first_step(config, dataset, contexts)
            failed = False
        finally:
            close_source(contexts, failed=failed)
        if report["first_step"]["status"] != "passed":
            problems.append("TigerGraph rejected every root of the first training batch")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_process_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    report["problems"] = problems
    report["status"] = "not_ready" if problems else "ready"
    report["graph_writes"] = 0
    return report
