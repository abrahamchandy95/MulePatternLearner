"""Publish aggregate behavioral pretraining results; never publish account IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=Path("artifacts/temporal/runs/zelle_forecast_20260918")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/experiments"))
    args = parser.parse_args()
    results = json.loads((args.runs / "results.json").read_text())
    tree = json.loads((args.runs / "rolling_trees/metrics.json").read_text())
    names = {
        "tabular": "Rolling-feature MLP",
        "rolling_trees": "Rolling-feature trees",
        "no_fourier": "Graph without Fourier",
        "temporal": "Graph with Fourier64",
    }
    summaries = []
    for variant in ("tabular", "rolling_trees", "no_fourier", "temporal"):
        runs = (
            [tree]
            if variant == "rolling_trees"
            else [r for r in results if r["config"]["variant"] == variant]
        )
        scores = np.array([r["metrics"]["test"]["average_precision"] for r in runs])
        summaries.append(
            {
                "variant": variant,
                "name": names[variant],
                "runs": len(runs),
                "test_ap_mean": float(scores.mean()),
                "test_ap_seed_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "validation_ap_mean": float(
                    np.mean([r["metrics"]["validation"]["average_precision"] for r in runs])
                ),
                "test_roc_auc_mean": float(
                    np.mean([r["metrics"]["test"]["roc_auc"] for r in runs])
                ),
                "test_precision_at_5pct_mean": float(
                    np.mean([r["metrics"]["test"]["precision_at_5pct"] for r in runs])
                ),
            }
        )
    selected = max(results, key=lambda r: r["metrics"]["validation"]["average_precision"])
    predictions = pd.read_parquet(
        args.runs
        / f"{selected['config']['variant']}_seed{selected['config']['seed']}_reveal1/test_predictions.parquet"
    )
    report: dict[str, Any] = {
        "experiment": "Self-supervised outgoing Zelle activity forecast, next 30 days",
        "not_mule_detection": True,
        "source": "Synthetic simulator export verified against live TigerGraph",
        "supervised_mule_experiments": "Pending target decision; no proxy labels silently substituted",
        "summaries": summaries,
        "validation_selected_model": {
            "variant": selected["config"]["variant"],
            "seed": selected["config"]["seed"],
            "test": selected["metrics"]["test"],
        },
        "test_unique_accounts": int(predictions["id"].nunique()),
        "test_unique_owner_groups": int(predictions["group_id"].nunique()),
        "test_account_cutoffs": len(predictions),
        "test_prevalence": float(predictions["target"].mean()),
        "neural_runs": results,
        "tree_baseline": tree,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "temporal_training_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(summaries))
    ax.bar(
        xs,
        [r["test_ap_mean"] for r in summaries],
        yerr=[r["test_ap_seed_std"] for r in summaries],
        capsize=5,
        color=["#8794a8", "#657c94", "#207f96", "#153d65"],
    )
    ax.axhline(
        report["test_prevalence"],
        color="#b26922",
        linestyle="--",
        label=f"Random ranking: {report['test_prevalence']:.3f}",
    )
    ax.set_xticks(
        xs, ["Rolling MLP", "Rolling trees", "Graph\nwithout Fourier", "Graph\nwith Fourier64"]
    )
    for x, r in zip(xs, summaries, strict=True):
        ax.text(
            float(x),
            r["test_ap_mean"] + 0.035,
            f"{r['test_ap_mean']:.3f}",
            ha="center",
            fontsize=11,
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out average precision")
    ax.set_title(
        "Future Zelle activity — behavioral pretraining", loc="left", fontweight="bold", pad=16
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0, 0.38))
    fig.text(
        0.125,
        0.02,
        "Synthetic data · 1,213 held-out accounts · 2 cutoffs · bars: mean ± seed SD\nThese results do not measure mule-account detection.",
        fontsize=9,
        color="#465161",
    )
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(args.output / "temporal_pretraining.png", dpi=180)
    plt.close(fig)
    table = "\n".join(
        f"| {r['name']} | {r['validation_ap_mean']:.3f} | {r['test_ap_mean']:.3f}{(' ± ' + format(r['test_ap_seed_std'], '.3f')) if r['runs'] > 1 else ''} | {r['test_roc_auc_mean']:.3f} | {r['test_precision_at_5pct_mean']:.3f} |"
        for r in summaries
    )
    content = f"""# Temporal training experiments — 18 September 2026

The temporal graph models are trained and their checkpoints are saved. **These
results measure future Zelle activity, not mule-account detection.** Supervised
mule masking experiments await a confirmed account target or an explicit decision
to use the simulator's laundering-intermediary proxy.

## Data and task

The verified export contains 2,950,984 payments (31,507 Zelle), 76,147 entity
vertices and 69,694 association tenures. The source has zero explicit mule-role
events and two laundering rings. Transfer fraud labels were never treated as
account mule labels. See [label audit](label_audit.json) and
[live verification](training_source_verification.json).

The automatic pretraining target is whether an internal deposit account sends
any Zelle payment in the next 30 days. Inputs contain only observations strictly
before its cutoff. Accounts and co-owner groups are disjoint across train,
validation and test. Cutoffs: April–September for training, October for validation,
and November/December for test. Test contains {report["test_unique_accounts"]:,}
unique accounts in {report["test_unique_owner_groups"]:,} owner groups,
{report["test_account_cutoffs"]:,} account-cutoff examples and 364 positives.
The random-ranking AP baseline is {report["test_prevalence"]:.3f}.

## Results

| Model | Validation AP | Test AP | Test AUROC | Precision at top 5% |
|---|---:|---:|---:|---:|
{table}

Neural entries report three seeds (42, 43, 44), with test AP mean ± sample standard
deviation. Trees are one deterministic histogram-boosted baseline: 50, 100 and 200
trees were compared on validation; {tree["selected_iterations"]} were selected.
All models use the same splits and cutoff-based entity features. The Fourier
ablation retains scalar recency/rolling features and timed sampling; it tests
removal of Fourier message inputs, not removal of every temporal signal.

![Behavioral pretraining comparison](temporal_pretraining.png)

Graph models substantially outperform these rolling-feature baselines on this
synthetic behavior task. Fourier64 adds only a small test increment, while its
validation AP is lower. There is no strong basis yet to claim that Fourier64
improves mule detection. The model selected by validation among the neural runs
is `{selected["config"]["variant"]}`, seed {selected["config"]["seed"]}; its test AP
is {selected["metrics"]["test"]["average_precision"]:.3f}. The temporal model was
not selected as champion by inspecting test results.

Each neural checkpoint was selected by validation AP and its operating threshold
by validation F1 before test scoring. Owner-group bootstrap intervals, per-cutoff
metrics, learning curves, configurations, code/data fingerprints and dependency
versions are in [the full aggregate results](temporal_training_results.json).
The intervals condition on this synthetic corpus; three initialization seeds do
not establish production reliability.

## Produced artifacts

- Nine neural checkpoints and one tree baseline under
  `artifacts/temporal/runs/zelle_forecast_20260918/`.
- A temporal checkpoint at `temporal_seed42_reveal1/model.pt`, using two 64D
  time inputs and a 32D learned entity representation.
- Scores and learned embeddings for all 7,969 internal deposit accounts at
  2025-01-01 under `artifacts/temporal/predictions/`. That forecast horizon is
  outside the extract and has not been evaluated.
- A separate nnPU trainer, nested account-label masking, whole-owner split
  isolation, dark-ring holdouts and per-ring recall reporting. The end-to-end
  nnPU path is tested on controlled fixtures; that test is not a detection result.

Raw staged data, account-level predictions and checkpoints stay in ignored local
artifact directories. This report contains aggregate results only. No training
run changed the live graph.

## What remains for mule detection

Choose confirmed mule supervision or explicitly accept the differently named
laundering-intermediary proxy. Then run the 10%, 25%, 50% and 100% reveal matrix
and separate dark-ring experiments, reporting actual revealed counts, prior
sensitivity, hidden-training recovery and held-out performance. Low-budget runs
with no available positives are skipped rather than having their masks changed.
Only two source rings and very few early positive roles limit this study even if
the proxy is selected; a larger independent-ring corpus is needed for a credible
claim of hidden-ring generalization.

Reproduction commands, label contract, sampling limits, research references and
clock assumptions are in [the training guide](../temporal_training.md).
"""
    (args.output / "temporal_training_results.md").write_text(content)
    print(
        json.dumps(
            {
                "summaries": summaries,
                "validation_selected": report["validation_selected_model"]["variant"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
