"""Print the reproducible feature ablation matrix without training or reading labels."""

import argparse
import json
from pathlib import Path

from mule_pattern_learner.temporal.live.config_schema import run_config
from mule_pattern_learner.temporal.live.contract import FeaturePlan
from mule_pattern_learner.temporal.live.experiments import feature_experiments
from mule_pattern_learner.temporal.live.model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional overrides of the built-in run")
    args = parser.parse_args()
    for name, config in feature_experiments(run_config(args.config)).items():
        plan = FeaturePlan.from_config(config)
        model = build_model(config, plan)
        print(
            json.dumps(
                {
                    "experiment": name,
                    "architecture": plan.architecture,
                    "groups": plan.groups,
                    "node_width": len(plan.node_names),
                    "message_width": len(plan.edge_names),
                    "parameters": sum(p.numel() for p in model.parameters()),
                }
            )
        )


if __name__ == "__main__":
    main()
