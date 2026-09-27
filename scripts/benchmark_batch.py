"""Build one real v5 training batch against TigerGraph without saving a model.

The batch is the first scheduled training step of the prepared dataset of the
built-in run (config.DEFAULT_CONFIG; main takes another run, as the golden-run test
passes), built the way train() builds it: the run's sampler (training-mode
resampling with the step seed), hub stubs from the dataset registry, and rejected
roots dropped. The report has REST calls and retries, seconds, stub and rejected
counts, the sampler backend and the digest of every batch tensor
(batching.assemble.tensor_digests, the definition the golden-run test pins). With
--train-step it also runs one optimizer step on the chosen device under the run's
determinism and reports its loss and objective, the first values train() logs. Two
code versions built the same batch and step when both print the same digests and
loss on one machine and device. Only read queries run; the live source is checked
with verify_frozen_source first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch

from mule_pattern_learner.batching.assemble import (
    batch_device,
    build_root_batch,
    tensor_digests,
    to_device,
)
from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig
from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.data.contexts import StreamingContextSource
from mule_pattern_learner.data.hub_registry import load_hub_registry
from mule_pattern_learner.data.manifest import dataset_mismatches, load_prepared
from mule_pattern_learner.data.observed_labels import load_observed_labels
from mule_pattern_learner.data.splits import sample_keys
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.model.loss import NonNegativePULoss
from mule_pattern_learner.paths import DatasetPaths
from mule_pattern_learner.pipeline.connect import open_context_source
from mule_pattern_learner.pipeline.prepare import find_datasets
from mule_pattern_learner.runtime.device import (
    choose_device,
    reserve_deterministic_cublas,
    torch_runtime,
)
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor
from mule_pattern_learner.training.objective import nnpu_objective, nnpu_step
from mule_pattern_learner.training.schedule import epoch_schedule
from mule_pattern_learner.training.trainer import build_optimizer, check_limits, training_samples


def rest_calls(source: Any) -> tuple[int, dict[str, int]]:
    """Successful REST calls and retries of the live executor (0 without one, e.g. a fake)."""
    fetcher = source.fetcher if isinstance(source, StreamingContextSource) else None
    executor = fetcher.executor if isinstance(fetcher, TigerGraphContextFetcher) else None
    if not isinstance(executor, TigerGraphExecutor):
        return 0, {}
    return executor.calls, dict(executor.retries)


def run_dataset(config: RunConfig) -> DatasetPaths:
    """The run's one dataset in data/; the benchmark only reads, so it never prepares one."""
    found = find_datasets(config)
    if len(found) != 1:
        raise ValueError(
            f"Found {len(found)} datasets of the run in data/; prepare one with "
            "`mule-temporal prepare`, or pass --dataset"
        )
    return found[0]


def main(config: RunConfig = DEFAULT_CONFIG) -> None:
    # Before any CUDA work: deterministic cuBLAS GEMMs need a fixed workspace.
    reserve_deterministic_cublas()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, help="prepared dataset (default: the run's dataset in data/)"
    )
    parser.add_argument("--output", type=Path, help="also write the report to this file")
    parser.add_argument("--device", help="override the configured device (auto, cpu, mps, cuda)")
    parser.add_argument(
        "--mode", choices=("train", "eval"), default="train", help="sampler mode of the batch"
    )
    parser.add_argument("--train-step", action="store_true", help="also run one optimizer step")
    args = parser.parse_args()
    dataset = run_dataset(config) if args.dataset is None else DatasetPaths(args.dataset)
    manifest, accounts = load_prepared(dataset)
    changed = dataset_mismatches(config, manifest)
    if changed:
        raise ValueError(f"Configuration differs from the prepared dataset: {changed}")
    plan = config.feature_plan()
    sampler, training, runtime = config.sampler, config.training, config.runtime
    check_limits(config, plan)
    device = choose_device(args.device or runtime.device)
    hubs = load_hub_registry(dataset, manifest)

    # The first step of epoch 0, exactly as train() schedules it.
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

    started = time.perf_counter()
    source = open_context_source(dataset, manifest, config)
    open_seconds = time.perf_counter() - started
    try:
        with torch_runtime(device, deterministic=runtime.deterministic, threads=runtime.threads):
            calls_before, _ = rest_calls(source)
            requests_before = source.query_calls
            started = time.perf_counter()
            prepared = build_root_batch(
                source,
                keys,
                fanouts=sampler.fanouts,
                device=batch_device(device),
                plan=plan,
                sampler=sampler,
                hubs=hubs,
                mode=args.mode,
                step_seed=step.seed,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            build_seconds = time.perf_counter() - started
            calls_after, retries = rest_calls(source)
            batch = prepared.batch
            report: dict[str, Any] = {
                "status": "passed" if batch is not None else "every_root_rejected",
                "device": str(device),
                "deterministic": runtime.deterministic,
                "mode": args.mode,
                "step_seed": step.seed,
                "roots": len(keys),
                "accepted_roots": int(prepared.accepted.sum()),
                "batch": prepared.stats,
                "rejections": dict(source.rejections),
                "context_requests": source.query_calls - requests_before,
                "rest_calls": calls_after - calls_before,
                "retries": retries,
                "source_open_seconds": open_seconds,
                "batch_seconds": build_seconds,
                "input_fingerprint": plan.fingerprint(),
                "sampler_fingerprint": sampler.fingerprint(),
                "purpose": "one_batch_readiness_not_model_quality_or_scale_proof",
            }
            if batch is not None:
                report["tensor_bytes"] = sum(t.numel() * t.element_size() for t in batch.values())
                report["tensor_digests"] = tensor_digests(batch)
            if args.train_step and batch is not None:
                report |= train_step(config, plan, device, batch, prepared, step)
    finally:
        source.close()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_process_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    if device.type == "mps":
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        report["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


def train_step(
    config: RunConfig,
    plan: FeaturePlan,
    device: torch.device,
    batch: dict[str, torch.Tensor],
    prepared: Any,
    step: Any,
) -> dict[str, Any]:
    """One nnPU optimizer step with the trainer's model, loss and seeding."""
    torch.manual_seed(config.training.seed)
    model = build_model(config.model, plan, config.sampler.fanouts[0]).to(device)
    optimizer = build_optimizer(model, config.training)
    prior, positive_weight = nnpu_objective(config.loss)
    loss_fn = NonNegativePULoss(prior=prior, positive_weight=positive_weight)
    started = time.perf_counter()
    # Rejected roots were dropped; the leading accepted rows are the positives.
    positives = int(prepared.accepted[: len(step.positives)].sum())
    loss = nnpu_step(
        model,
        optimizer,
        loss_fn,
        to_device(batch, device),
        positives,
        step.seed,
    )
    value = float(loss.value.cpu())  # waits for the accelerator
    if not np.isfinite(value):
        raise ValueError("Non-finite training loss in the benchmark step")
    return {
        "loss": value,
        # The unclamped risk; it differs from the loss when the nnPU correction fired.
        "objective": float(loss.objective.cpu()),
        "train_step_seconds": time.perf_counter() - started,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


if __name__ == "__main__":
    main()
