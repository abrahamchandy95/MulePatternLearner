"""What a bank knows of the mules it found before a cutoff, through read_supervision: the
only supervision training reads (docs/design.md, "Who may read what")."""

from __future__ import annotations

import pandas as pd

from ..tigergraph.executor import QueryExecutor
from ..tigergraph.queries import (
    READ_SUPERVISION,
    Phase,
    Supervision,
    parse_supervision,
    supervision_params,
)
from .frames import frame


def read_supervision(executor: QueryExecutor, cutoff_ms: int, phase: Phase) -> pd.DataFrame:
    """The mules of the customers phase sees that were discovered before cutoff_ms, one row
    each by account: account_id, discovered_ms, discovery_channel and mule_onset_ms.

    A row whose onset (the earliest suspicious activity) is 0 or after its discovery is
    refused: a mule is found only for what it has done, so the export is wrong.
    """
    rows = executor.run(READ_SUPERVISION, supervision_params(cutoff_ms, phase))
    found = sorted(parse_supervision(rows, cutoff_ms), key=lambda row: row.account_id)
    for row in found:
        if not 0 < row.mule_onset_ms <= row.discovered_ms:
            raise ValueError(
                f"{row.account_id} was discovered at {row.discovered_ms}, not at or after an "
                f"onset ({row.mule_onset_ms})"
            )
    return frame(Supervision, found)
