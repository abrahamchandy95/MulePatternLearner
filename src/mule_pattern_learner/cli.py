"""mule: find money-mule accounts with a temporal graph neural network on TigerGraph.

The code is being rebuilt to docs/design.md, so every command is still a stub: it says
so and exits with status 1.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

# The commands of the design ("Commands"), in the order a run uses them, with what each
# will do. None takes an option: worlds, roles, dates and settings live in config.
COMMANDS = {
    "install": "install the changed queries on TigerGraph",
    "parity": "compare every query with the reference graph and write the parity record",
    "prepare": "match the loaded world, check its calendar and labels, write the examples",
    "fetch": "fill the cache for every example the loaded world's roles need",
    "audit": "read the answer key and run the label checks, clock probe and feature audit",
    "train": "train the GNN, its ablations and the two XGBoost comparisons",
    "evaluate": "score the test world and write the metrics and the missed-mule report",
    "score": "score the configured accounts and dates with the best-selection model",
    "bench": "measure the cost of fetching, building and training on the loaded world",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mule", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, summary in COMMANDS.items():
        commands.add_parser(name, help=summary, description=f"Not built yet: {summary}.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run a command of the command line; every one is a stub that exits with status 1."""
    command: str = build_parser().parse_args(argv).command
    sys.exit(f"mule {command} is not built yet: the code is being rebuilt to docs/design.md")
