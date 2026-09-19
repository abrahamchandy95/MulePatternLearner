"""Qualify one configured nnPU batch against TigerGraph without saving a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from mule_pattern_learner.configuration import load_config
from mule_pattern_learner.device import choose_device
from mule_pattern_learner.temporal.live.batching import make_live_batch
from mule_pattern_learner.temporal.live.dataset import load_prepared, sample_keys
from mule_pattern_learner.temporal.live.memory import BatchLimits
from mule_pattern_learner.temporal.live.model import LiveTGAT
from mule_pattern_learner.temporal.live.pipeline import DEFAULT_CONFIG, dataset_path
from mule_pattern_learner.temporal.live.sampling import pu_batches
from mule_pattern_learner.temporal.live.source import open_context_source
from mule_pattern_learner.temporal.live.supervision import load_observed_labels, visible_labels
from mule_pattern_learner.training.loss import NonNegativePULoss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/temporal/batch_readiness.json")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = dataset_path(config)
    manifest, accounts = load_prepared(dataset)
    date = config["dates"]["train"][0]
    labels = visible_labels(load_observed_labels(accounts, dataset, manifest), date)
    train = accounts["split"].eq("train").to_numpy()
    marginal = np.flatnonzero(train & accounts["in_marginal"].to_numpy())
    positives = np.flatnonzero(train & labels)
    batch_size = int(config.get("batch_size", 64))
    fanouts = tuple(config.get("fanouts", [8, 4]))
    hidden = int(config.get("hidden", 64))
    BatchLimits().validate_model(batch_size, fanouts, hidden)
    device = choose_device(config.get("device"))
    torch.manual_seed(int(config.get("seed", 42)))
    torch.set_num_threads(int(config.get("threads", 4)))
    p, u = next(
        pu_batches(
            marginal,
            labels,
            np.random.default_rng(config.get("seed", 42)),
            batch_size,
            max_steps=1,
            positive_indices=positives,
        )
    )
    model = LiveTGAT(hidden, int(config.get("heads", 4)), float(config.get("dropout", 0.15))).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 0.001)))
    source = open_context_source(dataset, manifest)
    started = time.perf_counter()
    try:
        batch = make_live_batch(
            source,
            sample_keys(accounts.iloc[np.r_[p, u]], date, manifest),
            fanouts=fanouts,
            device=device,
        )
        fetch_seconds = time.perf_counter() - started
        logits = model(batch)
        targets = torch.zeros_like(logits)
        targets[: len(p)] = 1
        loss, _ = NonNegativePULoss(prior=float(config["class_prior"]))(logits, targets)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite qualification loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        loss_value = float(loss.detach().cpu())  # Wait for accelerator work.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = {
            "status": "passed",
            "device": str(device),
            "roots": len(p) + len(u),
            "unique_contexts": len(batch["x"]),
            "tensor_bytes": sum(t.numel() * t.element_size() for t in batch.values()),
            "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            "fetch_seconds": fetch_seconds,
            "total_seconds": time.perf_counter() - started,
            "database_calls": source.query_calls,
            "loss": loss_value,
            "purpose": "one_batch_readiness_not_model_quality_or_scale_proof",
        }
        if device.type == "mps":
            report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
            report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        source.close()


if __name__ == "__main__":
    main()
