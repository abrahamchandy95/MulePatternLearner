"""nnPU with observed-label selection over an injected temporal context source."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..encoding import BASIS_ID
from ..metrics import evaluate, select_threshold
from ..common import digest
from ..common import timestamp
from .batching import make_live_batch
from .contract import contract_fingerprint
from .dataset import load_prepared, sample_keys
from .model import LiveTGAT
from .source import ContextSource, open_context_source
from .sampling import pu_batches, evaluation_indices
from .supervision import label_summary, load_observed_labels, visible_labels
from .policy import validate_protocol
from .memory import BatchLimits
from mule_pattern_learner.device import choose_device
from mule_pattern_learner.training.loss import NonNegativePULoss


def output_paths(output: Path) -> tuple[Path, Path]:
    """A .pt output names the model; directory outputs retain the experiment API."""
    if output.suffix == ".pt":
        return output, output.with_name(output.stem + "_run")
    return output / "model.pt", output


def train(
    config: dict[str, Any],
    dataset: Path,
    output: Path,
    *,
    contexts: ContextSource | None = None,
) -> dict[str, Any]:
    validate_protocol(config)
    BatchLimits().validate_model(
        int(config.get("batch_size", 64)),
        tuple(config.get("fanouts", [8, 4])),
        int(config.get("hidden", 64)),
    )
    checkpoint_path, run_dir = output_paths(output)
    if checkpoint_path.exists() or run_dir.exists():
        raise FileExistsError(f"Experiment already exists: {output}")
    manifest, accounts = load_prepared(dataset)
    prepared = manifest["config"]
    for key in (
        "dates",
        "fanouts",
        "label_policy",
        "dataset_id",
        "per_relation",
        "evaluation_protocol",
        "scope_id",
    ):
        if config.get(key) != prepared.get(key):
            raise ValueError(f"Training setting differs from preparation: {key}")
    seed = int(config.get("seed", 42))
    prior = float(config["class_prior"])
    loss_function = NonNegativePULoss(prior=prior)
    device = choose_device(config.get("device"))
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("threads", 4)))
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    mask = load_observed_labels(accounts, dataset, manifest)
    samples: dict[str, list[dict[str, Any]]] = {split: [] for split in config["dates"]}
    for split, dates in config["dates"].items():
        for date in dates:
            eligible = accounts["split"].eq(split).to_numpy() & (
                accounts["first_seen_ts_ms"].to_numpy() < timestamp(date)
            )
            observed = visible_labels(mask, date)
            indices = np.flatnonzero(eligible)
            if split != "train":
                indices = evaluation_indices(
                    indices,
                    observed,
                    limit=config.get("evaluation_unlabeled_limit"),
                    seed=int(config.get("split_seed", 42)),
                )
            samples[split].append({"date": date, "indices": indices, "observed": observed})
    usable = [s for s in samples["train"] if s["observed"][s["indices"]].any()]
    validation_labels = np.concatenate([s["observed"][s["indices"]] for s in samples["validation"]])
    if not usable or len(np.unique(validation_labels)) != 2:
        raise ValueError(
            "Training needs revealed positives and validation needs both observed classes; "
            "masks/splits were not changed"
        )
    run_dir.mkdir(parents=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    mask.to_parquet(run_dir / "observed_labels.parquet", index=False)
    print(
        json.dumps(
            {
                "device": str(device),
                "known_mules": label_summary(mask),
                "loss": "nnPU",
                "model": str(checkpoint_path),
            }
        ),
        flush=True,
    )
    store = contexts or open_context_source(dataset, manifest)
    model = LiveTGAT(
        int(config.get("hidden", 64)),
        int(config.get("heads", 4)),
        float(config.get("dropout", 0.15)),
        config.get("variant", "temporal"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 0.001)), weight_decay=0.0001
    )
    fanouts = tuple(config.get("fanouts", [8, 4]))
    batch_size = int(config.get("batch_size", 64))
    started = time.perf_counter()

    def progress(**fields: Any) -> None:
        print(
            json.dumps(
                {
                    **fields,
                    "query_calls": store.query_calls,
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }
            ),
            flush=True,
        )

    def batch(indices: np.ndarray, date: str) -> dict[str, torch.Tensor]:
        return make_live_batch(
            store,
            sample_keys(accounts.iloc[indices], date, manifest),
            fanouts=fanouts,
            device=device,
        )

    def scores(split: str) -> pd.DataFrame:
        # Only observed labels are attached here. Oracle evaluation is a separate command.
        model.eval()
        frames = []
        with torch.inference_mode():
            for sample in samples[split]:
                idx = sample["indices"]
                values = []
                progress(evaluating=split, date=sample["date"], accounts=0, total=len(idx))
                for start in range(0, len(idx), batch_size):
                    values.extend(
                        torch.sigmoid(model(batch(idx[start : start + batch_size], sample["date"])))
                        .cpu()
                        .tolist()
                    )
                    progress(
                        evaluating=split,
                        date=sample["date"],
                        accounts=min(start + batch_size, len(idx)),
                        total=len(idx),
                    )
                frame = accounts.iloc[idx][["account_id", "group_id"]].copy()
                frame["date"] = sample["date"]
                frame["observed_label"] = sample["observed"][idx].astype(np.int64)
                frame["score"] = values
                frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    best_ap, best_epoch = -1.0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []
    try:
        for epoch in range(int(config.get("epochs", 30))):
            model.train()
            losses = []
            progress(training="train", epoch=epoch + 1, step=0)
            for sample in usable:
                for positive, marginal in pu_batches(
                    sample["indices"][
                        accounts.iloc[sample["indices"]]
                        .get("in_marginal", pd.Series(True, index=sample["indices"]))
                        .to_numpy(bool)
                    ],
                    sample["observed"],
                    rng,
                    batch_size,
                    max_steps=config.get("steps_per_epoch"),
                    positive_indices=sample["indices"][sample["observed"][sample["indices"]]],
                ):
                    logits = model(batch(np.r_[positive, marginal], sample["date"]))
                    targets = torch.zeros_like(logits)
                    targets[: len(positive)] = 1
                    loss, _ = loss_function(logits, targets)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    progress(
                        training="train",
                        epoch=epoch + 1,
                        step=len(losses),
                        date=sample["date"],
                        loss=losses[-1],
                    )
            validation = scores("validation")
            metrics = evaluate(
                validation["observed_label"].to_numpy(), validation["score"].to_numpy(), 0.5
            )
            ap = metrics["average_precision"]
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(np.mean(losses)),
                    "steps": len(losses),
                    "validation_proxy_ap": ap,
                }
            )
            print(json.dumps(history[-1]), flush=True)
            if ap > best_ap:
                best_ap, best_epoch = ap, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch + 1 - best_epoch >= int(config.get("patience", 6)):
                break
        model.load_state_dict(best_state)
        validation = scores("validation")
        threshold = select_threshold(
            validation["observed_label"].to_numpy(), validation["score"].to_numpy()
        )
        selection = evaluate(
            validation["observed_label"].to_numpy(), validation["score"].to_numpy(), threshold
        )
        # Save the selected checkpoint; oracle truth is not a training dependency.
        torch.save(
            {
                "state_dict": best_state,
                "config": config,
                "basis_id": BASIS_ID,
                "contract": contract_fingerprint(),
                "dataset_manifest_sha256": digest(dataset / "manifest.json"),
                "threshold": threshold,
                "feature_dim": model.node[0].in_features,
                "selected_on": "validation_observed_label_proxy_ap",
                "training_protocol": "scoped_observed_label_nnpu_v4",
                "evaluation_protocol": config["evaluation_protocol"],
                "known_mules": label_summary(mask),
                "training_device": str(device),
            },
            checkpoint_path,
        )
        results = {}
        # Model outputs contain observed labels only. Oracle evaluation is a separate command.
        for split, frame in (("validation", validation), ("test", scores("test"))):
            frame.to_parquet(run_dir / (split + "_predictions.parquet"), index=False)
            results[split] = evaluate(
                frame["observed_label"].to_numpy(), frame["score"].to_numpy(), threshold
            )
        result = {
            "status": "complete",
            "cohort": manifest["cohort"],
            "label_policy": config["label_policy"],
            "variant": model.variant,
            "seed": seed,
            "known_mules": label_summary(mask),
            "device": str(device),
            "loss": "nnPU",
            "class_prior": prior,
            "revealed_training_accounts": label_summary(mask)["train"],
            "best_epoch": best_epoch,
            "history": history,
            "observed_label_proxy": results,
            "evaluation_protocol": config["evaluation_protocol"],
            "validation_proxy": selection,
            "checkpoint": str(checkpoint_path),
            "database_calls_during_training": store.query_calls,
            "performance_claim": config["evaluation_protocol"] + "_observed_label_proxy_only",
            "evaluation_unlabeled_limit": config.get("evaluation_unlabeled_limit"),
        }
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    finally:
        store.close()
