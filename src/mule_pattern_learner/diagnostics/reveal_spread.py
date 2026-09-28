"""How the label reveal's outcome varies with its salt, replayed offline.

The reveal (tigergraph.reveal) ran once on the graph, with the configured salt: which
mules a bank would have discovered by each split's cutoff, and which of those it
revealed, depends on that salt's draws. Its Python mirror (reference.label_reveal.plan,
with the job's own hash and defaults) replays it for many salts on the reveal's inputs,
which RevealInputReader reads once from the graph, read-only; the job itself never runs.
Per salt and split: the mules, those discovered before the cutoff (eligible) and those
revealed, which the reveal's budget caps. The spread says how much of a run's labels is
the salt's luck.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

import pandas as pd

from ..artifacts import DIAGNOSTIC_TABLES
from ..contract.graph_schema import PHASE_SPLIT
from ..reference.label_reveal import counts_by_split, plan

COLUMNS = DIAGNOSTIC_TABLES["reveal_spread"]
# The salts replayed: 0 to 999, which hold the configured salt of the built-in run (42).
SALTS = range(1000)


class RevealInputReader(Protocol):
    """The reveal's inputs in a scope, as its inputs query prints them (read-only)."""

    def read(self, scope_id: str) -> list[dict[str, Any]]: ...


def reveal_spread(
    inputs: list[dict[str, Any]], params: Mapping[str, Any], salts: Iterable[int] = SALTS
) -> pd.DataFrame:
    """The reveal spread table: mules, eligible and revealed per salt and split.

    ``params`` are the reveal's parameters (tigergraph.reveal.reveal_parameters with
    apply off); each salt replaces its salt.
    """
    records: list[tuple[int, str, str, int]] = []
    for salt in salts:
        result = plan(inputs, {**params, "salt": salt})
        mules = {part: 0 for part in PHASE_SPLIT}
        for mule in result["mules"].values():
            if mule["part"] in mules:
                mules[mule["part"]] += 1
        eligible = counts_by_split(result, "eligible")
        revealed = counts_by_split(result, "revealed")
        for part, split in PHASE_SPLIT.items():
            records += [
                (salt, split, "mules", mules[part]),
                (salt, split, "eligible", eligible[part]),
                (salt, split, "revealed", revealed[part]),
            ]
    return pd.DataFrame(records, columns=list(COLUMNS))
