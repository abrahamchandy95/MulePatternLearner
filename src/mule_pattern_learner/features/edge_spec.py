"""Dimensional contract for HAS_PAID edge features.

The single source of truth for how wide a per-edge feature vector is: the fixed
scalar fields, the per-bin channel count, and the flattened width a fixed-edge-dim
GNN layer expects. This is pure arithmetic over the schema -- no tensors, no I/O,
no torch -- so the database-client layer (tigergraph.derivation) can derive
edge_dim from a graph-read bin count without pulling in the model stack, and the
edge-feature builder (features.temporal) and the PyG transform both agree with it
by construction.
"""


class EdgeFeatureError(ValueError):
    pass


SCALAR_FEATURE_NAMES: tuple[str, ...] = (
    "amount_transferred",
    "transaction_count",
    "active_span_days",
    "active_bin_count",
    "recency_days",
    "duration_days",
    "active_bin_fraction",
    "amount_per_transaction",
)

NUM_SCALAR_FEATURES: int = len(SCALAR_FEATURE_NAMES)
NUM_BIN_CHANNELS: int = 2


def flat_edge_dim(max_bins: int) -> int:
    """Width of the flattened per-edge feature vector for a given bin count."""
    if max_bins < 1:
        raise EdgeFeatureError(f"max_bins must be >= 1, got {max_bins}")
    return NUM_SCALAR_FEATURES + max_bins * NUM_BIN_CHANNELS
