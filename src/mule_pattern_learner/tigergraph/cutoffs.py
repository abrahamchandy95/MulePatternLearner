"""The read-only cutoff query: the last event visible at each cutoff time."""

from __future__ import annotations

from ..contract.server import CUTOFF_QUERY
from .executor import QueryExecutor, checked_rows, printed


class TigerGraphCutoffs:
    """The CutoffReader of data.ports: the cutoff query (contract.server.CUTOFF_QUERY)."""

    def __init__(self, executor: QueryExecutor) -> None:
        self.executor = executor

    def last_visible_seqs(self, cutoff_times: list[int]) -> dict[int, int]:
        """The sequence of the last event or entity visible at each cutoff time."""
        rows = checked_rows(
            self.executor.run(CUTOFF_QUERY, {"cutoff_times": cutoff_times}, timeout_s=900.0)
        )
        clocks = printed(rows, "last_visible_seqs")
        return {ms: int(clocks[str(ms)]) for ms in cutoff_times}
