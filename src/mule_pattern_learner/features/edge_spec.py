"""
Feature-width definitions for HAS_PAID edges.

Sets the dimension of each edge's feature vector: 8 scalar features plus 2
binned features (amount, count) unrolled across max_bins time-slices.
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
    """Width of the per-edge feature vector per bin count."""
    if max_bins < 1:
        raise EdgeFeatureError(f"max_bins must be >= 1, got {max_bins}")
    return NUM_SCALAR_FEATURES + max_bins * NUM_BIN_CHANNELS
