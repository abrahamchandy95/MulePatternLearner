"""Reproducible temporal training with external supervision and locked tests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
from typing import Any

from .common import timestamp as timestamp

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
from torch import nn

from .encoding import BASIS_ID
from .metrics import evaluate, grouped_ap_interval, select_threshold
from .model import Normalizer, TemporalModel, make_batch, nnpu_loss
from .snapshots import Snapshot
from .staging import digest
from .supervision import (
    account_groups,
    forecast_targets,
    labels_at_cutoff,
    read_labels,
    graph_reveal_mask,
    reveal_mask,
    split_accounts,
)


def predict(
    model: TemporalModel,
    snapshot: Snapshot,
    roots: NDArray[np.int64],
    normalizer: Normalizer,
    batch_size: int,
    fanouts: tuple[int, int],
    device: str,
) -> NDArray[np.float64]:
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(roots), batch_size):
            batch = make_batch(
                snapshot,
                roots[start : start + batch_size],
                normalizer,
                variant=model.variant,
                fanouts=fanouts,
                device=device,
            )
            values.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(values).astype(np.float64) if values else np.array([], dtype=np.float64)


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Experiment directory already exists: {output}")
    started = time.monotonic()
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("threads", 4)))
    torch.use_deterministic_algorithms(True)
    stage = Path(config["stage"])
    snapshots_dir = Path(config["snapshots"])
    task = config["task"]
    if task not in {"mule_pu", "zelle_forecast"}:
        raise ValueError("Unsupported task")
    nodes = pd.read_parquet(stage / "nodes.parquet")
    associations = pd.read_parquet(stage / "associations.parquet")
    labels, label_metadata = (
        read_labels(Path(config["labels"]), nodes) if task == "mule_pu" else (None, {})
    )
    dark_rings = set(int(i) for i in config.get("dark_ring_ids", []))
    forced_groups: set[str] = set()
    if dark_rings:
        if labels is None or min(dark_rings) < 0:
            raise ValueError("Dark rings require label memberships; zero is a valid ring ID")
        groups = account_groups(nodes, associations)
        forced_groups = {
            groups[int(i)] for i in labels.loc[labels["ring_id"].isin(dark_rings), "node_index"]
        }
    accounts = split_accounts(
        nodes, associations, seed=int(config.get("split_seed", 42)), test_groups=forced_groups
    )
    mask_source = config.get("mask_source", "resample")
    if mask_source == "graph":
        if labels is None or float(config.get("reveal_fraction", 1)) != 1:
            raise ValueError(
                "Graph-mask mode requires mule labels and reveal_fraction=1; the stored mask sets the budget"
            )
        revealed = graph_reveal_mask(accounts, labels)
    elif mask_source == "resample":
        revealed = reveal_mask(accounts, float(config.get("reveal_fraction", 1)), seed)
    else:
        raise ValueError("mask_source must be graph or resample")
    dates_by_split: dict[str, list[str]] = config["dates"]
    if not all(dates_by_split.get(key) for key in ("train", "validation", "test")):
        raise ValueError("All chronological splits are required")
    if (
        not max(dates_by_split["train"])
        < min(dates_by_split["validation"])
        <= max(dates_by_split["validation"])
        < min(dates_by_split["test"])
    ):
        raise ValueError("Cutoffs must be chronological and disjoint")
    horizon = int(config.get("horizon_days", 30)) * 86_400_000
    if task == "zelle_forecast":
        if timestamp(max(dates_by_split["train"])) + horizon > timestamp(
            min(dates_by_split["validation"])
        ) or timestamp(max(dates_by_split["validation"])) + horizon > timestamp(
            min(dates_by_split["test"])
        ):
            raise ValueError("Forecast targets cross the next split's time boundary")
    samples: dict[str, list[dict[str, Any]]] = {key: [] for key in dates_by_split}
    snapshot_hashes = {}
    for split, dates in dates_by_split.items():
        for date in dates:
            snapshot_path = snapshots_dir / date
            snapshot = Snapshot.load(snapshot_path)
            cutoff = timestamp(date)
            if snapshot.metadata["cutoff_ms"] != cutoff or snapshot.metadata[
                "stage_sha256"
            ] != digest(stage / "manifest.json"):
                raise ValueError("Snapshot provenance mismatch")
            snapshot_hashes[date] = digest(snapshot_path / "metadata.json")
            if task == "mule_pu":
                assert labels is not None
                truth, positive = labels_at_cutoff(accounts, labels, cutoff, revealed)
            else:
                truth = forecast_targets(stage, accounts, cutoff, horizon)
                positive = truth == 1
            eligible = accounts["split"].eq(split).to_numpy() & (
                accounts["first_seen_ts_ms"].to_numpy() < cutoff
            )
            if split != "train":
                eligible &= truth >= 0
            indices = np.flatnonzero(eligible)
            samples[split].append(
                {
                    "date": date,
                    "snapshot": snapshot,
                    "indices": indices,
                    "truth": truth,
                    "positive": positive,
                }
            )
    usable_train = [
        s
        for s in samples["train"]
        if len(s["indices"]) and (task != "mule_pu" or s["positive"][s["indices"]].any())
    ]
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    accounts.assign(reveal_selected=revealed).to_parquet(output / "split_mask.parquet", index=False)
    if not usable_train:
        result = {
            "status": "skipped",
            "reason": "No available revealed positives in any training cutoff; mask was not changed",
            "config": config,
        }
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    validation_truth = np.concatenate([s["truth"][s["indices"]] for s in samples["validation"]])
    if len(np.unique(validation_truth)) != 2:
        result = {
            "status": "skipped",
            "reason": "Validation needs positive and negative ground truth for checkpoint selection; the split was not changed",
            "config": config,
        }
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    normalizer = Normalizer.fit(samples["train"][-1]["snapshot"])
    device = str(config.get("device", "cpu"))
    model = TemporalModel(
        len(normalizer.mean),
        int(config.get("hidden", 32)),
        float(config.get("dropout", 0.15)),
        str(config["variant"]),
    ).to(device)
    if config.get("pretrained_checkpoint"):
        checkpoint = torch.load(
            config["pretrained_checkpoint"], map_location=device, weights_only=True
        )
        if checkpoint["basis_id"] != BASIS_ID or checkpoint["variant"] != model.variant:
            raise ValueError("Incompatible pretraining checkpoint")
        pre_config = checkpoint["config"]
        consulted_until = timestamp(max(pre_config["dates"]["validation"]))
        if pre_config["task"] == "zelle_forecast":
            consulted_until += int(pre_config.get("horizon_days", 30)) * 86_400_000
        if consulted_until > timestamp(min(dates_by_split["validation"])):
            raise ValueError(
                "Pretraining selection used information after the fine-tuning validation boundary"
            )
        if checkpoint["provenance"]["stage_sha256"] != digest(stage / "manifest.json"):
            raise ValueError("Pretraining dataset differs from fine-tuning dataset")
        # Restore normalization along with the encoder; reset the task-specific head.
        normalizer = Normalizer(
            checkpoint["normalizer_mean"].numpy(), checkpoint["normalizer_scale"].numpy()
        )
        model.load_state_dict(
            {
                **model.state_dict(),
                **{
                    key: value
                    for key, value in checkpoint["state_dict"].items()
                    if not key.startswith("head.")
                },
            }
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 0.001)), weight_decay=0.0001
    )
    fanouts = tuple(int(i) for i in config.get("fanouts", [8, 4]))
    if len(fanouts) != 2:
        raise ValueError("Exactly two fanouts are required")
    fanout_pair = (fanouts[0], fanouts[1])
    batch_size = int(config.get("batch_size", 256))
    rng = np.random.default_rng(seed)
    history = []
    best_ap = -1.0
    best_state = deepcopy(model.state_dict())
    best_epoch = 0

    def score_split(split: str) -> pd.DataFrame:
        frames = []
        for sample in samples[split]:
            idx = sample["indices"]
            roots = accounts.iloc[idx]["node_index"].to_numpy(np.int64)
            score = predict(
                model, sample["snapshot"], roots, normalizer, batch_size, fanout_pair, device
            )
            frame = accounts.iloc[idx][["id", "group_id", "node_index"]].copy()
            frame["cutoff"] = sample["date"]
            frame["target"] = sample["truth"][idx]
            frame["score"] = score
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    for epoch in range(int(config.get("epochs", 20))):
        model.train()
        losses = []
        for _ in range(int(config.get("steps_per_epoch", 20))):
            sample = usable_train[int(rng.integers(len(usable_train)))]
            idx = sample["indices"]
            marginal = rng.choice(idx, size=batch_size, replace=True)
            if task == "mule_pu":
                positives = idx[sample["positive"][idx]]
                pos = rng.choice(positives, size=min(64, batch_size), replace=True)
                selected = np.r_[pos, marginal]
            else:
                selected = marginal
            roots = accounts.iloc[selected]["node_index"].to_numpy(np.int64)
            batch = make_batch(
                sample["snapshot"],
                roots,
                normalizer,
                variant=model.variant,
                fanouts=fanout_pair,
                device=device,
            )
            logits = model(batch)
            if task == "mule_pu":
                loss = nnpu_loss(
                    logits[:-batch_size], logits[-batch_size:], float(config["class_prior"])
                )
            else:
                target = torch.from_numpy(sample["truth"][selected].astype(np.float32)).to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = score_split("validation")
        vm = evaluate(
            validation["target"].to_numpy(np.int64), validation["score"].to_numpy(np.float64), 0.5
        )
        ap = vm["average_precision"]
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_ap": ap})
        print(json.dumps({"variant": model.variant, "seed": seed, **history[-1]}), flush=True)
        if ap is not None and ap > best_ap:
            best_ap, best_epoch = ap, epoch + 1
            best_state = deepcopy(model.state_dict())
        if epoch + 1 - best_epoch >= int(config.get("patience", 5)):
            break
    model.load_state_dict(best_state)
    validation = score_split("validation")
    threshold = select_threshold(
        validation["target"].to_numpy(np.int64), validation["score"].to_numpy(np.float64)
    )
    # First test evaluation occurs after checkpoint and threshold selection.
    test = score_split("test")
    evaluation = {}
    for split, frame in (("validation", validation), ("test", test)):
        y = frame["target"].to_numpy(np.int64)
        scores = frame["score"].to_numpy(np.float64)
        evaluation[split] = evaluate(y, scores, threshold)
        evaluation[split]["ap_group_bootstrap_95pct"] = grouped_ap_interval(
            y, scores, frame["group_id"].to_numpy()
        )
        evaluation[split]["by_cutoff"] = {
            str(date): evaluate(
                group["target"].to_numpy(np.int64), group["score"].to_numpy(np.float64), threshold
            )
            for date, group in frame.groupby("cutoff")
        }
        frame.to_parquet(output / f"{split}_predictions.parquet", index=False)
    recovery = []
    if task == "mule_pu":
        assert labels is not None
        rings = test["id"].map(labels.set_index("account_id")["ring_id"]).to_numpy()
        test_truth = test["target"].to_numpy(np.int64)
        test_scores = test["score"].to_numpy(np.float64)
        test_ids = test["id"].to_numpy()
        evaluation["test"]["ring_recall"] = {
            str(int(ring)): {
                "dark": int(ring) in dark_rings,
                "positive_account_cutoffs": int(((rings == ring) & (test_truth == 1)).sum()),
                "unique_positive_accounts": int(
                    len(np.unique(test_ids[(rings == ring) & (test_truth == 1)]))
                ),
                "recall": float(
                    (test_scores[(rings == ring) & (test_truth == 1)] >= threshold).mean()
                ),
            }
            for ring in np.unique(rings[(rings >= 0) & (test_truth == 1)])
        }
        for sample in samples["train"]:
            idx = sample["indices"]
            # Explicitly a training-population diagnostic, never reported as test lift.
            eligible = (sample["truth"][idx] >= 0) & ~sample["positive"][idx]
            idx = idx[eligible]
            score = predict(
                model,
                sample["snapshot"],
                accounts.iloc[idx]["node_index"].to_numpy(np.int64),
                normalizer,
                batch_size,
                fanout_pair,
                device,
            )
            recovery.append(
                {"cutoff": sample["date"], **evaluate(sample["truth"][idx], score, threshold)}
            )
    counts = {
        split: [
            {
                "cutoff": s["date"],
                "accounts": len(s["indices"]),
                "true_positives": int((s["truth"][s["indices"]] == 1).sum()),
                "revealed_positive_accounts": int(s["positive"][s["indices"]].sum())
                if split == "train" and task == "mule_pu"
                else None,
            }
            for s in values
        ]
        for split, values in samples.items()
    }
    provenance = {
        "stage_sha256": digest(stage / "manifest.json"),
        "snapshot_sha256": snapshot_hashes,
        "split_mask_sha256": digest(output / "split_mask.parquet"),
        "basis_id": BASIS_ID,
        "labels_sha256": digest(Path(config["labels"])) if task == "mule_pu" else None,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "code_sha256": {
            path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "versions": {
            name: version(name) for name in ("torch", "numpy", "pandas", "scikit-learn", "pyarrow")
        },
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "normalizer_mean": torch.from_numpy(normalizer.mean),
            "normalizer_scale": torch.from_numpy(normalizer.scale),
            "variant": model.variant,
            "basis_id": BASIS_ID,
            "config": config,
            "provenance": provenance,
            "threshold": threshold,
            "best_epoch": best_epoch,
        },
        output / "model.pt",
    )
    result = {
        "status": "complete",
        "task": task,
        "target_definition": label_metadata.get("target_definition", "outgoing_zelle_next_30_days"),
        "config": config,
        "provenance": provenance,
        "best_epoch": best_epoch,
        "counts": counts,
        "history": history,
        "metrics": evaluation,
        "hidden_training_recovery": recovery,
        "label_metadata": label_metadata,
        "seconds": time.monotonic() - started,
        "limitations": [
            "Synthetic source data",
            "Known-time clock absent",
            "Transductive topology; held-out labels are never model inputs",
            "Account/group holdouts plus chronological cutoffs",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
