"""Build one real v5 training batch against TigerGraph without saving a model.

The batch is the first scheduled training step of the prepared dataset, built the
way train() builds it: the configured sampler (training-mode resampling with the
step seed), hub stubs from the dataset registry, and rejected roots dropped. The
report has REST calls and retries, seconds, stub and rejected counts and the
sampler backend. With --train-step it also runs one optimizer step on the chosen
device under the configured determinism. Only read queries run; the live source is
checked with verify_frozen_source first.
"""

from __future__ import annotations

import os

# Deterministic cuBLAS GEMMs need this before CUDA initializes; a user value wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mule_pattern_learner.device import choose_device, torch_runtime  # noqa: E402
from mule_pattern_learner.temporal.live.batching import build_root_batch  # noqa: E402
from mule_pattern_learner.temporal.live.contract import FeaturePlan, SamplerPlan  # noqa: E402
from mule_pattern_learner.temporal.live.dataset import (  # noqa: E402
    load_prepared,
    preparation_mismatches,
    sample_keys,
)
from mule_pattern_learner.temporal.live.executor import TigerGraphExecutor  # noqa: E402
from mule_pattern_learner.temporal.live.hubs import load_hub_registry  # noqa: E402
from mule_pattern_learner.temporal.live.model import build_model  # noqa: E402
from mule_pattern_learner.temporal.live.config_schema import run_config  # noqa: E402
from mule_pattern_learner.temporal.live.pipeline import dataset_path, prepared_config  # noqa: E402
from mule_pattern_learner.temporal.live.sampling import epoch_schedule  # noqa: E402
from mule_pattern_learner.temporal.live.source import open_context_source  # noqa: E402
from mule_pattern_learner.temporal.live.supervision import load_observed_labels  # noqa: E402
from mule_pattern_learner.temporal.live.training import (  # noqa: E402
    RunSettings,
    build_optimizer,
    nnpu_objective,
    nnpu_step,
    training_samples,
)
from mule_pattern_learner.temporal.loss import NonNegativePULoss  # noqa: E402


def rest_calls(source: Any) -> tuple[int, dict[str, int]]:
    """Successful REST calls and retries of the live executor (0 without one, e.g. SQLite)."""
    executor = getattr(source, "executor", None)
    if not isinstance(executor, TigerGraphExecutor):
        return 0, {}
    return executor.calls, dict(executor.retries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional overrides of the built-in run")
    parser.add_argument("--dataset", type=Path, help="prepared dataset (default: from config)")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/temporal/batch_readiness.json")
    )
    parser.add_argument("--device", help="override the configured device (auto, cpu, mps, cuda)")
    parser.add_argument(
        "--mode", choices=("train", "eval"), default="train", help="sampler mode of the batch"
    )
    parser.add_argument("--train-step", action="store_true", help="also run one optimizer step")
    args = parser.parse_args()
    config = run_config(args.config)
    dataset = args.dataset or dataset_path(config)
    manifest, accounts = load_prepared(dataset)
    config = prepared_config(config, manifest)
    changed = preparation_mismatches(config, manifest)
    if changed:
        raise ValueError(f"Configuration differs from the prepared dataset: {changed}")
    plan, sampler = FeaturePlan.from_config(config), SamplerPlan.from_config(config)
    settings = RunSettings.from_config(config)
    settings.check_limits(plan, sampler)
    device = choose_device(args.device or settings.device)
    hubs = load_hub_registry(dataset, manifest)

    # The first step of epoch 0, exactly as train() schedules it.
    samples = training_samples(config, accounts, load_observed_labels(accounts, dataset, manifest))
    if not samples:
        raise ValueError("No train cutoff has revealed positives; train() would refuse too")
    step = epoch_schedule(
        samples,
        np.random.default_rng(settings.seed),
        settings.batch_size,
        epoch=0,
        seed=settings.seed,
        max_steps=1,
    )[0]
    keys = sample_keys(accounts.iloc[step.indices], step.date, manifest)

    started = time.perf_counter()
    source = open_context_source(dataset, manifest, config)
    open_seconds = time.perf_counter() - started
    batch_device = device if device.type == "cuda" else torch.device("cpu")
    try:
        with torch_runtime(device, deterministic=settings.deterministic, threads=settings.threads):
            calls_before, _ = rest_calls(source)
            requests_before = source.query_calls
            started = time.perf_counter()
            prepared = build_root_batch(
                source,
                keys,
                fanouts=settings.fanouts,
                device=batch_device,
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
                "deterministic": settings.deterministic,
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
            if args.train_step and batch is not None:
                report |= train_step(config, settings, plan, device, batch, prepared, step)
    finally:
        source.close()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_process_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    if device.type == "mps":
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        report["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


def train_step(
    config: dict[str, Any],
    settings: RunSettings,
    plan: FeaturePlan,
    device: torch.device,
    batch: dict[str, torch.Tensor],
    prepared: Any,
    step: Any,
) -> dict[str, Any]:
    """One nnPU optimizer step with the trainer's model, loss and seeding."""
    torch.manual_seed(settings.seed)
    model = build_model(config, plan).to(device)
    optimizer = build_optimizer(model, config)
    prior, positive_weight = nnpu_objective(config)
    loss_fn = NonNegativePULoss(prior=prior, positive_weight=positive_weight)
    started = time.perf_counter()
    # Rejected roots were dropped; the leading accepted rows are the positives.
    positives = int(prepared.accepted[: len(step.positives)].sum())
    loss = nnpu_step(
        model,
        optimizer,
        loss_fn,
        {k: v.to(device) for k, v in batch.items()},
        positives,
        step.seed,
    )
    value = float(loss.cpu())  # waits for the accelerator
    if not np.isfinite(value):
        raise ValueError("Non-finite training loss in the benchmark step")
    return {
        "loss": value,
        "train_step_seconds": time.perf_counter() - started,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


if __name__ == "__main__":
    main()
