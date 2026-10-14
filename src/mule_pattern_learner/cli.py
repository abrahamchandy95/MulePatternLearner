"""mule: find money-mule accounts with a temporal graph neural network on TigerGraph.

The code is being rebuilt to docs/design.md, so every command is still a stub: it says
so and exits with status 1.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

# The commands of the design, in the order a run uses them, with what each will do.
COMMANDS = {
    "install": "install the queries on TigerGraph",
    "prepare": "write the customer hold-out and the weekly example grid",
    "fetch": "fetch and cache the subgraph of every example",
    "train": "train the GNN and the rankers it is compared with",
    "evaluate": "score every weekly test date and report the metrics",
    "score": "score accounts at a date with the saved model",
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
