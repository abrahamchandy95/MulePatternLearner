from dataclasses import dataclass
from typing import cast

from mule_pattern_learner.features.edge_spec import flat_edge_dim
from mule_pattern_learner.tigergraph.client import Client

_DERIVE_QUERY_NAME = "derive_max_bins"
_REFERENCE_EPOCH_QUERY_NAME = "derive_reference_epoch"


class TemporalDerivationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GraphTemporalSpec:
    """Immutable, derived description of the graph's temporal edge features.

    This is the single source of truth for the HAS_PAID bin width within a run.
    It is produced once by derive_temporal_spec() and then passed to both the
    edge-feature builder (padding width) and the model (edge_dim), so the two
    can never disagree. It holds no database handle and performs no I/O.
    """

    max_bins: int

    def __post_init__(self) -> None:
        if self.max_bins < 1:
            raise TemporalDerivationError(f"max_bins must be >= 1, got {self.max_bins}")

    @property
    def edge_dim(self) -> int:
        """Flattened per-edge feature width a fixed-edge-dim GNN layer expects."""
        return flat_edge_dim(self.max_bins)


def _scalar_int(raw: list[object], key: str) -> int:
    for block in raw:
        if isinstance(block, dict) and key in block:
            value = cast(dict[str, object], block)[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TemporalDerivationError(
                    f"{_DERIVE_QUERY_NAME}: {key!r} was not an int: {value!r}"
                )
            return value
    raise TemporalDerivationError(f"{_DERIVE_QUERY_NAME}: key {key!r} not found in output: {raw!r}")


def derive_temporal_spec(client: Client) -> GraphTemporalSpec:
    """Run derive_max_bins against the graph and build a GraphTemporalSpec.

    Performs a full-edge scan on the server (the derive_max_bins query) and
    verifies that the declared num_bins attribute agrees with the actual
    amount_bins / count_bins list lengths; a mismatch means padding would
    silently truncate real data, so it is raised rather than tolerated.
    """
    raw = cast(list[object], client.conn.runInstalledQuery(_DERIVE_QUERY_NAME, {}))
    max_bins = _scalar_int(raw, "max_bins")
    amount_len = _scalar_int(raw, "max_amount_len")
    count_len = _scalar_int(raw, "max_count_len")

    if not (max_bins == amount_len == count_len):
        raise TemporalDerivationError(
            "inconsistent bin counts in graph data: "
            + f"num_bins={max_bins}, amount_bins.size()={amount_len}, "
            + f"count_bins.size()={count_len}. The declared num_bins disagrees "
            + "with actual list lengths; padding would truncate real temporal data."
        )

    return GraphTemporalSpec(max_bins=max_bins)


def derive_reference_epoch_s(client: Client) -> float:
    """Run derive_reference_epoch against the graph; return the snapshot epoch.

    The reference epoch is the latest transaction time in the graph (unix
    seconds), the snapshot moment recency edge-features are measured against.
    Deriving it keeps the loader consistent with the data and avoids a "now"
    that post-dates it. Performs a full HAS_PAID scan on the server.
    """
    raw = cast(list[object], client.conn.runInstalledQuery(_REFERENCE_EPOCH_QUERY_NAME, {}))
    return float(_scalar_int(raw, "reference_epoch_s"))
