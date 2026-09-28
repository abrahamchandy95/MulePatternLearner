"""Train, audit and compare the control experiments of the built-in run.

python scripts/run_experiments.py [SUITE or VARIANT ...] runs the named suites and
variants (the controls suite when none is named), the baseline always among them, with
the seeds 42, 43 and 44. Every variant is validated offline and the run matrix printed
with a time bound before training. Complete runs are kept, runs whose settings differ
move to results/archive/ and train again, and the comparison tables, figures and
report.md are always rewritten under results/experiments/<suite>/. It prints one JSON
result and exits 1 unless every run is trained and audited.
"""

import argparse
import json
import sys

from mule_pattern_learner.experiments.variants import describe, select


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=describe(), formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("names", nargs="*", metavar="SUITE or VARIANT", help="suites or variants")
    args = parser.parse_args()
    try:
        select(args.names)
    except ValueError as error:
        parser.error(str(error))
    # Loaded only now, so --help needs no torch; the workspace comes before any CUDA work.
    from mule_pattern_learner.runtime.device import reserve_deterministic_cublas

    reserve_deterministic_cublas()
    from mule_pattern_learner.experiments.runner import run_suite

    result = run_suite(args.names)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
