"""Regenerate gsql/temporal/training_context.gsql after editing its shared contract."""

import argparse
from pathlib import Path
import sys

from mule_pattern_learner.temporal.live.queries import render_context_query


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only compare the rendered text with the file; exit 1 when it differs",
    )
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[2] / "gsql/temporal/training_context.gsql"
    text = render_context_query()
    if args.check:
        same = path.read_text() == text
        print(f"{path.name} {'matches' if same else 'differs from'} the generator")
        return 0 if same else 1
    path.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
