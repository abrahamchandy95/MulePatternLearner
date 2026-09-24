"""Print the reproducible feature ablation matrix without training or reading labels."""

import argparse
import json
from pathlib import Path

from mule_pattern_learner.configuration import load_config
from mule_pattern_learner.temporal.live.contract import FeaturePlan
from mule_pattern_learner.temporal.live.experiments import feature_experiments
from mule_pattern_learner.temporal.live.model import LiveTGAT
from mule_pattern_learner.temporal.live.pipeline import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    for name, config in feature_experiments(load_config(args.config, live=True)).items():
        plan = FeaturePlan.from_config(config)
        model = LiveTGAT(int(config.get("hidden", 64)), int(config.get("heads", 4)), plan=plan)
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
