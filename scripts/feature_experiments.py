"""Print the reproducible feature ablation matrix without training or reading labels."""

import argparse
import json
from pathlib import Path

from mule_pattern_learner.config import run_config
from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.experiments.variants import feature_experiments
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.runtime.device import reserve_deterministic_cublas


def main():
    # Before any CUDA work: deterministic cuBLAS GEMMs need a fixed workspace.
    reserve_deterministic_cublas()
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
