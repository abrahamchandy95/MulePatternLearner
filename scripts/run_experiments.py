"""Train, audit and compare the control experiments of the built-in run.

python scripts/run_experiments.py [SUITE or VARIANT ...] runs the named suites and
variants (the controls suite when none is named), the baseline always among them, with
the ten seeds 42 to 51. Every variant is validated offline and the run matrix printed
before training, with a range of hours estimated from the suite's finished runs (a cold
first run per new seed, cached runs after it) beside an upper bound. Complete runs are
kept, so a suite trained with fewer seeds trains only the new ones; runs whose settings
differ move to results/archive/ and train again, and the comparison tables, figures and
report.md are always rewritten under results/experiments/<suite>/. It shows the run
matrix, a line for each run as it finishes and then the top of the comparison; the
full records are in the suite's and the runs' files. It exits 1 unless every run is
trained and audited.
"""

import argparse
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
    from mule_pattern_learner.cli import record_error, stopped
    from mule_pattern_learner.experiments.runner import run_suite, suite_summary
    from mule_pattern_learner.runtime.console import end_progress, show
    from mule_pattern_learner.tigergraph.executor import TransientQueryError

    try:
        result = run_suite(args.names)
    except TransientQueryError as error:
        # An outage while preparing the dataset stops the suite before any run; its
        # record goes to the dataset's or the suite's events.jsonl, as a command's does.
        end_progress()
        record_error("run_experiments.py", error, None)
        raise SystemExit(stopped("run_experiments.py", error)) from None
    except Exception as error:
        end_progress()
        record_error("run_experiments.py", error, None)
        raise
    except BaseException:
        end_progress()
        raise
    # The summary replaces a line rewritten in place, such as the scoring of an audit.
    show(suite_summary(result))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
