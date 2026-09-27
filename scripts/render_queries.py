"""Regenerate gsql/queries/training_context.gsql after editing its shared contract."""

import argparse
import sys

from mule_pattern_learner.contract.server import CONTEXT_QUERY_FILE
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.tigergraph.render import render_context_query


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only compare the rendered text with the file; exit 1 when it differs",
    )
    args = parser.parse_args()
    path = GSQL_DIR / CONTEXT_QUERY_FILE
    text = render_context_query()
    if args.check:
        same = path.read_text() == text
        print(f"{path.name} {'matches' if same else 'differs from'} the generator")
        return 0 if same else 1
    path.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
