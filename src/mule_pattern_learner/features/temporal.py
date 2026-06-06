import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

import torch
from torch import Tensor

from mule_pattern_learner.features.edge_spec import (
    NUM_BIN_CHANNELS,
    NUM_SCALAR_FEATURES,
    EdgeFeatureError,
)

_SECONDS_PER_DAY: float = 86400.0


def _string_keyed(mapping: dict[object, object]) -> dict[str, object]:
    return {str(k): v for k, v in mapping.items()}


_NEAR_CAP_WARN_FRACTION: float = 0.9


def log1p_compress(value: float) -> float:
    """Compress a heavy-tailed non-negative quantity via log(1 + x).

    Monetary amounts and transaction counts span many orders of magnitude;
    raw values let the largest edges dominate. log(1 + x) compresses that
    range, and the (1 + x) shift maps 0 -> 0 (a bin with no activity stays
    zero) while remaining defined at x = 0. Negative inputs are clamped to 0.
    """
    return math.log1p(max(value, 0.0))


@dataclass(frozen=True, slots=True)
class EdgeFeatures:
    src_ids: tuple[str, ...]
    dst_ids: tuple[str, ...]
    scalar_feats: Tensor
    bin_seq: Tensor
    max_bins: int

    @property
    def num_edges(self) -> int:
        return len(self.src_ids)

    def __post_init__(self) -> None:
        if self.max_bins < 1:
            raise EdgeFeatureError(f"max_bins must be >= 1, got {self.max_bins}")
        e = len(self.src_ids)
        if len(self.dst_ids) != e:
            raise EdgeFeatureError(f"src/dst length mismatch: {e} vs {len(self.dst_ids)}")
        if tuple(self.scalar_feats.shape) != (e, NUM_SCALAR_FEATURES):
            raise EdgeFeatureError(
                f"scalar_feats shape {tuple(self.scalar_feats.shape)} "
                + f"!= ({e}, {NUM_SCALAR_FEATURES})"
            )
        if tuple(self.bin_seq.shape) != (e, self.max_bins, NUM_BIN_CHANNELS):
            raise EdgeFeatureError(
                f"bin_seq shape {tuple(self.bin_seq.shape)} "
                + f"!= ({e}, {self.max_bins}, {NUM_BIN_CHANNELS})"
            )


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise EdgeFeatureError(f"{field}: expected number, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    raise EdgeFeatureError(f"{field}: expected number, got {type(value).__name__}")


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise EdgeFeatureError(f"{field}: expected int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise EdgeFeatureError(f"{field}: expected int, got {type(value).__name__}")


_DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def _as_epoch_seconds(value: object, field: str) -> float:
    """Parse a DATETIME field into epoch seconds (UTC).

    TigerGraph serializes DATETIME differently across configurations: an
    installed-query path may return epoch-second integers, while other setups
    return a formatted "YYYY-MM-DD HH:MM:SS" string. Both are accepted; strings
    are interpreted as UTC so the result is deterministic regardless of the
    machine's local timezone.
    """
    if isinstance(value, bool):
        raise EdgeFeatureError(f"{field}: expected datetime, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.strptime(value, _DATETIME_FORMAT)
        except ValueError as exc:
            raise EdgeFeatureError(
                f"{field}: could not parse datetime string {value!r} "
                + f"with format {_DATETIME_FORMAT!r}"
            ) from exc
        return parsed.replace(tzinfo=timezone.utc).timestamp()
    raise EdgeFeatureError(
        f"{field}: expected epoch number or datetime string, " + f"got {type(value).__name__}"
    )


def _attributes_of(edge: object) -> dict[str, object]:
    if not isinstance(edge, dict):
        raise EdgeFeatureError(f"edge is not a dict: {edge!r}")
    edge_typed = _string_keyed(cast(dict[object, object], edge))
    attrs = edge_typed.get("attributes")
    if not isinstance(attrs, dict):
        raise EdgeFeatureError(f"edge missing attributes dict: {edge_typed!r}")
    return _string_keyed(cast(dict[object, object], attrs))


@dataclass(frozen=True, slots=True)
class _RawScalars:
    total_amount: float
    total_num_txns: float
    span_days: float
    num_bins: int
    first_epoch_s: float
    last_epoch_s: float


@dataclass(frozen=True, slots=True)
class _TimeDeltas:
    recency_days: float
    duration_days: float


def _extract_endpoints(edge: dict[str, object]) -> tuple[str, str]:
    """Pull the source and destination account ids from an edge object."""
    src = edge.get("from_id")
    dst = edge.get("to_id")
    if not isinstance(src, str) or not src:
        raise EdgeFeatureError(f"edge missing 'from_id': {edge!r}")
    if not isinstance(dst, str) or not dst:
        raise EdgeFeatureError(f"edge missing 'to_id': {edge!r}")
    return src, dst


def _parse_scalar_fields(attrs: dict[str, object]) -> _RawScalars:
    """Extract and type-check the raw scalar attributes of one edge."""
    return _RawScalars(
        total_amount=_as_float(attrs.get("total_amount"), "total_amount"),
        total_num_txns=_as_float(attrs.get("total_num_txns"), "total_num_txns"),
        span_days=_as_float(attrs.get("span_days"), "span_days"),
        num_bins=_as_int(attrs.get("num_bins"), "num_bins"),
        first_epoch_s=_as_epoch_seconds(attrs.get("first_txn_date"), "first_txn_date"),
        last_epoch_s=_as_epoch_seconds(attrs.get("last_txn_date"), "last_txn_date"),
    )


def _compute_time_deltas(raw: _RawScalars, reference_epoch_s: float) -> _TimeDeltas:
    """Convert epoch-second timestamps into day-scale recency and duration.

    recency_days is measured from a caller-supplied reference time (the
    prediction/snapshot moment), never 'now', so the transform is leakage-safe.
    """
    recency_days = max((reference_epoch_s - raw.last_epoch_s) / _SECONDS_PER_DAY, 0.0)
    duration_days = max((raw.last_epoch_s - raw.first_epoch_s) / _SECONDS_PER_DAY, 0.0)
    return _TimeDeltas(recency_days=recency_days, duration_days=duration_days)


def _pad_bins(raw: object, field: str, max_bins: int) -> tuple[list[float], int]:
    """Pad or truncate a bin list to exactly max_bins; report the raw length."""
    if not isinstance(raw, list):
        raise EdgeFeatureError(f"{field}: expected list, got {type(raw).__name__}: {raw!r}")
    raw_list: list[object] = list(cast(list[object], raw))
    raw_len = len(raw_list)
    out: list[float] = [0.0] * max_bins
    n = min(raw_len, max_bins)
    for i in range(n):
        out[i] = _as_float(raw_list[i], f"{field}[{i}]")
    return out, raw_len


def _count_active_bins(count_bins: list[float]) -> int:
    """Count time bins that contain at least one transaction (count > 0).

    This is the true per-edge temporal footprint, derived from the data, not
    the global num_bins attribute (which is identical on every edge and
    therefore carries no per-edge signal).
    """
    return sum(1 for c in count_bins if c > 0.0)


def _build_scalar_row(
    raw: _RawScalars, deltas: _TimeDeltas, active_bin_count: int, max_bins: int
) -> list[float]:
    """Assemble the per-edge scalar feature vector in SCALAR_FEATURE_NAMES order."""
    active_bin_fraction = active_bin_count / max_bins if max_bins > 0 else 0.0
    amount_per_transaction = (
        raw.total_amount / raw.total_num_txns if raw.total_num_txns > 0.0 else 0.0
    )
    row: list[float] = [
        log1p_compress(raw.total_amount),
        log1p_compress(raw.total_num_txns),
        raw.span_days,
        float(active_bin_count),
        deltas.recency_days,
        deltas.duration_days,
        active_bin_fraction,
        amount_per_transaction,
    ]
    if len(row) != NUM_SCALAR_FEATURES:
        raise EdgeFeatureError(
            f"internal: built {len(row)} scalars, expected {NUM_SCALAR_FEATURES}"
        )
    return row


def _build_bin_row(amount_bins: list[float], count_bins: list[float], max_bins: int) -> list[float]:
    """Interleave (compressed amount, count) per bin into a flat row of 2*max_bins."""
    interleaved: list[float] = []
    for i in range(max_bins):
        interleaved.append(log1p_compress(amount_bins[i]))
        interleaved.append(max(count_bins[i], 0.0))
    return interleaved


def _warn_on_bin_capacity(
    max_raw_bins: int, truncated_count: int, total_edges: int, max_bins: int
) -> None:
    if truncated_count > 0:
        warnings.warn(
            f"{truncated_count}/{total_edges} edges had more than "
            + f"max_bins={max_bins} and were SILENTLY TRUNCATED "
            + f"(largest seen: {max_raw_bins}). Real temporal data was dropped. "
            + "Re-derive max_bins from the data and rebuild.",
            stacklevel=2,
        )
        return
    near_cap_floor = int(_NEAR_CAP_WARN_FRACTION * max_bins)
    if max_bins > 0 and near_cap_floor <= max_raw_bins < max_bins:
        warnings.warn(
            f"Largest edge had {max_raw_bins} bins, approaching max_bins={max_bins}. "
            + "Consider re-deriving the width before bins exceed it.",
            stacklevel=2,
        )


def build_edge_features(
    edges: Sequence[object],
    reference_epoch_s: float,
    max_bins: int,
) -> EdgeFeatures:
    if max_bins < 1:
        raise EdgeFeatureError(f"max_bins must be >= 1, got {max_bins}")

    src_ids: list[str] = []
    dst_ids: list[str] = []
    scalar_rows: list[float] = []
    bin_rows: list[float] = []

    max_raw_bins = 0
    truncated_count = 0

    for edge in edges:
        if not isinstance(edge, dict):
            raise EdgeFeatureError(f"edge is not a dict: {edge!r}")
        edge_typed = _string_keyed(cast(dict[object, object], edge))
        attrs = _attributes_of(edge_typed)

        src, dst = _extract_endpoints(edge_typed)
        raw = _parse_scalar_fields(attrs)
        deltas = _compute_time_deltas(raw, reference_epoch_s)

        amount_bins, amount_len = _pad_bins(attrs.get("amount_bins"), "amount_bins", max_bins)
        count_bins, count_len = _pad_bins(attrs.get("count_bins"), "count_bins", max_bins)
        raw_bin_len = max(amount_len, count_len)

        active_bin_count = _count_active_bins(count_bins)

        src_ids.append(src)
        dst_ids.append(dst)
        scalar_rows.extend(_build_scalar_row(raw, deltas, active_bin_count, max_bins))
        bin_rows.extend(_build_bin_row(amount_bins, count_bins, max_bins))

        if raw_bin_len > max_raw_bins:
            max_raw_bins = raw_bin_len
        if raw_bin_len > max_bins:
            truncated_count += 1

    _warn_on_bin_capacity(max_raw_bins, truncated_count, len(src_ids), max_bins)

    e = len(src_ids)
    scalar_feats = torch.tensor(scalar_rows, dtype=torch.float32).reshape(e, NUM_SCALAR_FEATURES)
    bin_seq = torch.tensor(bin_rows, dtype=torch.float32).reshape(e, max_bins, NUM_BIN_CHANNELS)

    return EdgeFeatures(
        src_ids=tuple(src_ids),
        dst_ids=tuple(dst_ids),
        scalar_feats=scalar_feats,
        bin_seq=bin_seq,
        max_bins=max_bins,
    )


def flatten_bin_seq(bin_seq: Tensor) -> Tensor:
    """Collapse a [E, bins, channels] sequence into a flat [E, bins*channels]."""
    if bin_seq.dim() != 3:
        raise EdgeFeatureError(
            f"bin_seq must be 3D [E, bins, channels], got {tuple(bin_seq.shape)}"
        )
    e = bin_seq.shape[0]
    return bin_seq.reshape(e, -1)


def flat_edge_features(features: EdgeFeatures) -> Tensor:
    """Concatenate scalar features and flattened bins into one [E, k] matrix.

    Some GNN layers (those taking a fixed edge feature dimension) require edge
    attributes as a flat 2D tensor rather than a 3D sequence. k == flat_edge_dim.
    """
    return torch.cat([features.scalar_feats, flatten_bin_seq(features.bin_seq)], dim=1)
