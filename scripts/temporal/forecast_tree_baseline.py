"""Strong rolling-feature tree baseline; select tree count on chronological validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mule_pattern_learner.configuration import load_config
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from mule_pattern_learner.temporal.metrics import evaluate, grouped_ap_interval, select_threshold
from mule_pattern_learner.temporal.snapshots import Snapshot
from mule_pattern_learner.temporal.staging import digest
from mule_pattern_learner.temporal.supervision import forecast_targets, split_accounts
from mule_pattern_learner.temporal.training import timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/temporal/zelle_forecast.toml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    config = load_config(args.config)
    if config["task"] != "zelle_forecast":
        raise ValueError(
            "This baseline uses automatic behavior targets, not unlabeled accounts as negatives"
        )
    stage = Path(config["stage"])
    nodes = pd.read_parquet(stage / "nodes.parquet")
    associations = pd.read_parquet(stage / "associations.parquet")
    accounts = split_accounts(nodes, associations, seed=config["split_seed"])
    features = {}
    frames = {}
    for split, dates in config["dates"].items():
        xs = []
        rows = []
        for date in dates:
            cutoff = timestamp(date)
            snapshot = Snapshot.load(Path(config["snapshots"]) / date)
            targets = forecast_targets(stage, accounts, cutoff, config["horizon_days"] * 86_400_000)
            mask = (accounts["split"] == split) & (accounts["first_seen_ts_ms"] < cutoff)
            frame = accounts.loc[mask].copy()
            frame["cutoff"] = date
            frame["target"] = targets[mask.to_numpy()]
            xs.append(snapshot.x[frame["node_index"].to_numpy(np.int64)])
            rows.append(frame)
        features[split] = np.concatenate(xs)
        frames[split] = pd.concat(rows, ignore_index=True)
    args.output.mkdir(parents=True)
    plan: dict[str, Any] = {
        "candidate_tree_counts": [50, 100, 200],
        "max_leaf_nodes": 15,
        "learning_rate": 0.05,
        "l2_regularization": 1.0,
        "random_seed": 42,
        "early_stopping": False,
        "selection": "Chronological validation AP; test only after selecting tree count",
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    candidates = []
    best = None
    best_ap = -1.0
    for iterations in plan["candidate_tree_counts"]:
        model = HistGradientBoostingClassifier(
            max_iter=iterations,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        )
        model.set_params(early_stopping=False)
        model.fit(features["train"], frames["train"]["target"].to_numpy())
        scores = model.predict_proba(features["validation"])[:, 1]
        metric = evaluate(frames["validation"]["target"].to_numpy(np.int64), scores, 0.5)
        candidates.append({"iterations": iterations, "validation_ap": metric["average_precision"]})
        if metric["average_precision"] > best_ap:
            best_ap = metric["average_precision"]
            best = model
    assert best is not None
    validation_scores = best.predict_proba(features["validation"])[:, 1]
    threshold = select_threshold(
        frames["validation"]["target"].to_numpy(np.int64), validation_scores
    )
    metrics = {}
    for split in ("validation", "test"):
        scores = best.predict_proba(features[split])[:, 1]
        y = frames[split]["target"].to_numpy(np.int64)
        metrics[split] = evaluate(y, scores, threshold)
        metrics[split]["ap_group_bootstrap_95pct"] = grouped_ap_interval(
            y, scores, frames[split]["group_id"].to_numpy()
        )
        frames[split].assign(score=scores).to_parquet(
            args.output / f"{split}_predictions.parquet", index=False
        )
    joblib.dump(
        {"model": best, "threshold": threshold, "config": config}, args.output / "model.joblib"
    )
    result = {
        "task": "zelle_forecast",
        "variant": "rolling_trees",
        "plan": plan,
        "candidates": candidates,
        "selected_iterations": best.max_iter,
        "metrics": metrics,
        "stage_sha256": digest(stage / "manifest.json"),
        "config": config,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
