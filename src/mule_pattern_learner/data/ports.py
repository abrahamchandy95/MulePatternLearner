"""The read ports of the data layer, owned by the code that reads through them.

Preparation, the context source, scoring and the ground-truth audit read the graph only
through these protocols. The tigergraph adapters satisfy them structurally, and only
the pipeline builds those adapters. Ground truth has its own port,
evaluation.truth.TruthReader, so it is not even on training's import surface.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import pandas as pd

    from ..contract.feature_groups import FeaturePlan
    from ..contract.graph_schema import ContextKey
    from ..contract.sampler_plan import SamplerPlan
    from .hub_registry import HubRegistry


class ContextFetcher(Protocol):
    """Runs the context request of one block of keys (1 to contract.bounds.REQUEST_KEYS).

    It returns validated rows in key order, a status row where the graph rejected a
    key, and the REST calls the request took. Checks it ran are counted in
    ``diagnostics``.
    """

    def request(
        self,
        keys: list[ContextKey],
        *,
        plan: FeaturePlan,
        sampler: SamplerPlan,
        hop: int,
        emit_encodings: bool,
        diagnostics: Counter[str],
    ) -> tuple[list[dict[str, Any]], int]: ...


class ScopeReader(Protocol):
    """The accounts of the graph's split, in pages in account order, with observed labels."""

    def population_pages(self) -> Iterable[list[dict[str, Any]]]: ...


class CutoffReader(Protocol):
    """The last event sequence visible at each cutoff time (in milliseconds)."""

    def last_visible_seqs(self, cutoff_times: list[int]) -> dict[int, int]: ...


class HubReader(Protocol):
    """The hub registry of root cutoffs: per phase of the split by_phase, else phase 0."""

    def hub_registry(
        self, cutoff_seqs: Iterable[int], *, threshold: int, by_phase: bool = False
    ) -> HubRegistry: ...


class ObservedLabelReader(Protocol):
    """Known positives and their discovery times; zero means unlabeled.

    The labels are those revealed in the graph, which the scope population reports;
    ``metadata`` holds its rows.
    """

    def read(self, metadata: pd.DataFrame) -> pd.DataFrame: ...
