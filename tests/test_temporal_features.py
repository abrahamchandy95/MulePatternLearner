from __future__ import annotations

import math
import warnings
from typing import cast

import pytest

from mule_pattern_learner.tigergraph.temporal import (
    NUM_BIN_CHANNELS,
    NUM_SCALAR_FEATURES,
    SCALAR_FEATURE_NAMES,
    EdgeFeatureError,
    EdgeFeatures,
    build_edge_features,
    flat_edge_dim,
    flat_edge_features,
    flatten_bin_seq,
    log1p_compress,
)

_FIRST_EPOCH_S = 1_735_689_600
_BIN_SECONDS = 14 * 86_400
_LAST_EPOCH_S = _FIRST_EPOCH_S + _BIN_SECONDS
_REFERENCE_EPOCH_S = float(_FIRST_EPOCH_S + 180 * 86_400)


def _make_edge(
    amount_bins: object,
    count_bins: object,
    *,
    src: str = "A1",
    dst: str = "A2",
    total_amount: float = 22.99,
    total_num_txns: float = 1.0,
    span_days: float = 14.0,
    num_bins: int | None = None,
    first_epoch_s: int | str = _FIRST_EPOCH_S,
    last_epoch_s: int | str = _LAST_EPOCH_S,
) -> dict[str, object]:
    if num_bins is None:
        num_bins = len(cast(list[object], count_bins)) if isinstance(count_bins, list) else 0
    return {
        "e_type": "HAS_PAID",
        "from_id": src,
        "to_id": dst,
        "attributes": {
            "total_amount": total_amount,
            "total_num_txns": total_num_txns,
            "span_days": span_days,
            "num_bins": num_bins,
            "first_txn_date": first_epoch_s,
            "last_txn_date": last_epoch_s,
            "amount_bins": amount_bins,
            "count_bins": count_bins,
        },
    }


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)


def _as_number(value: object) -> float:
    assert isinstance(value, (int, float))
    return float(value)


def _scalar(features: EdgeFeatures, edge_index: int, name: str) -> float:
    rows = cast(list[list[object]], features.scalar_feats.tolist())
    row = rows[edge_index]
    return _as_number(row[SCALAR_FEATURE_NAMES.index(name)])


def _bin_pair(features: EdgeFeatures, edge_index: int, bin_index: int) -> tuple[float, float]:
    seq = cast(list[list[list[object]]], features.bin_seq.tolist())
    pair = seq[edge_index][bin_index]
    return _as_number(pair[0]), _as_number(pair[1])


class TestLog1pCompress:
    def test_matches_math_log1p(self) -> None:
        assert _close(log1p_compress(22.99), math.log1p(22.99))

    def test_zero_maps_to_zero(self) -> None:
        assert log1p_compress(0.0) == 0.0

    def test_negative_is_clamped(self) -> None:
        assert log1p_compress(-5.0) == 0.0


class TestFlatEdgeDim:
    def test_width_formula(self) -> None:
        assert flat_edge_dim(13) == NUM_SCALAR_FEATURES + 13 * NUM_BIN_CHANNELS

    def test_rejects_zero(self) -> None:
        with pytest.raises(EdgeFeatureError):
            _ = flat_edge_dim(0)


class TestShapes:
    def test_scalar_and_bin_shapes(self) -> None:
        edge = _make_edge([22.99] + [0.0] * 12, [1] + [0] * 12)
        feats = build_edge_features([edge, edge], _REFERENCE_EPOCH_S, max_bins=13)
        assert feats.num_edges == 2
        assert tuple(feats.scalar_feats.shape) == (2, NUM_SCALAR_FEATURES)
        assert tuple(feats.bin_seq.shape) == (2, 13, NUM_BIN_CHANNELS)
        assert feats.max_bins == 13

    def test_derived_width_threads_through(self) -> None:
        edge = _make_edge([1.0] * 13, [1] * 13)
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=26)
        assert tuple(feats.bin_seq.shape) == (1, 26, NUM_BIN_CHANNELS)

    def test_endpoints_preserved(self) -> None:
        edge = _make_edge([1.0], [1], src="ACC_X", dst="ACC_Y")
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=4)
        assert feats.src_ids == ("ACC_X",)
        assert feats.dst_ids == ("ACC_Y",)


class TestScalarValues:
    def test_amount_transferred_is_compressed(self) -> None:
        edge = _make_edge([22.99], [1], total_amount=22.99)
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=4)
        assert _close(_scalar(feats, 0, "amount_transferred"), math.log1p(22.99))

    def test_recency_is_relative_to_reference(self) -> None:
        edge = _make_edge([1.0], [1])
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=4)
        assert _close(_scalar(feats, 0, "recency_days"), 166.0)

    def test_duration_spans_first_to_last(self) -> None:
        edge = _make_edge([1.0], [1])
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=4)
        assert _close(_scalar(feats, 0, "duration_days"), 14.0)

    def test_amount_per_transaction(self) -> None:
        edge = _make_edge([1.0], [1], total_amount=100.0, total_num_txns=4.0)
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=4)
        assert _close(_scalar(feats, 0, "amount_per_transaction"), 25.0)

    def test_amount_per_transaction_zero_when_no_txns(self) -> None:
        edge = _make_edge([0.0], [0], total_amount=0.0, total_num_txns=0.0)
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=4)
        assert _scalar(feats, 0, "amount_per_transaction") == 0.0


class TestBinSequence:
    def test_first_bin_channels(self) -> None:
        edge = _make_edge([22.99] + [0.0] * 12, [1] + [0] * 12)
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=13)
        amount0, count0 = _bin_pair(feats, 0, 0)
        assert _close(amount0, math.log1p(22.99))
        assert _close(count0, 1.0)

    def test_padding_is_zero(self) -> None:
        edge = _make_edge([5.0], [2])
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=8)
        for j in range(1, 8):
            amount_j, count_j = _bin_pair(feats, 0, j)
            assert amount_j == 0.0
            assert count_j == 0.0


class TestFlatten:
    def test_flat_edge_features_width(self) -> None:
        edge = _make_edge([1.0] * 13, [1] * 13)
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=13)
        flat = flat_edge_features(feats)
        assert tuple(flat.shape) == (1, flat_edge_dim(13))

    def test_flatten_bin_seq_collapses_last_dims(self) -> None:
        edge = _make_edge([1.0] * 5, [1] * 5)
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=5)
        flat = flatten_bin_seq(feats.bin_seq)
        assert tuple(flat.shape) == (1, 5 * NUM_BIN_CHANNELS)


class TestTruncationGuard:
    def test_truncation_warns_once(self) -> None:
        edge = _make_edge([float(i) for i in range(40)], list(range(40)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = build_edge_features([edge, edge], _REFERENCE_EPOCH_S, max_bins=13)
        assert len(caught) == 1
        message = str(caught[0].message)
        assert "TRUNCATED" in message
        assert "max_bins=13" in message

    def test_no_warning_when_within_capacity(self) -> None:
        edge = _make_edge([1.0], [1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=100)
        assert len(caught) == 0

    def test_at_cap_is_silent(self) -> None:
        edge = _make_edge([1.0] * 13, [1] * 13)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=13)
        assert len(caught) == 0

    def test_just_below_cap_warns(self) -> None:
        edge = _make_edge([1.0] * 19, [1] * 19)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=20)
        assert len(caught) == 1
        assert "approaching" in str(caught[0].message)


class TestTimestampFormats:
    def test_string_timestamps_match_epoch_ints(self) -> None:
        first_str = "2025-01-01 00:00:00"
        last_str = "2025-01-15 00:00:00"
        first_int = 1_735_689_600
        last_int = first_int + 14 * 86_400

        edge_str = _make_edge([22.99], [1], first_epoch_s=first_str, last_epoch_s=last_str)
        edge_int = _make_edge([22.99], [1], first_epoch_s=first_int, last_epoch_s=last_int)
        feats_str = build_edge_features([edge_str], _REFERENCE_EPOCH_S, max_bins=4)
        feats_int = build_edge_features([edge_int], _REFERENCE_EPOCH_S, max_bins=4)

        assert _close(
            _scalar(feats_str, 0, "recency_days"),
            _scalar(feats_int, 0, "recency_days"),
        )
        assert _close(
            _scalar(feats_str, 0, "duration_days"),
            _scalar(feats_int, 0, "duration_days"),
        )

    def test_string_timestamp_duration(self) -> None:
        edge = _make_edge(
            [22.99],
            [1],
            first_epoch_s="2025-01-01 00:00:00",
            last_epoch_s="2025-01-15 00:00:00",
        )
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=4)
        assert _close(_scalar(feats, 0, "duration_days"), 14.0)


class TestActiveBinCount:
    def test_single_active_bin(self) -> None:
        amount_bins = [0.0] * 6 + [208.26] + [0.0] * 6
        count_bins = [0] * 6 + [1] + [0] * 6
        edge = _make_edge(amount_bins, count_bins)
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=13)
        assert _close(_scalar(feats, 0, "active_bin_count"), 1.0)
        assert _close(_scalar(feats, 0, "active_bin_fraction"), 1.0 / 13.0)

    def test_two_active_bins(self) -> None:
        amount_bins = [0.0, 10.0, 0.0, 20.0] + [0.0] * 9
        count_bins = [0, 1, 0, 1] + [0] * 9
        edge = _make_edge(amount_bins, count_bins)
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=13)
        assert _close(_scalar(feats, 0, "active_bin_count"), 2.0)

    def test_ignores_num_bins_attribute(self) -> None:
        # num_bins attribute says 13, but only 1 bin actually has activity.
        amount_bins = [5.0] + [0.0] * 12
        count_bins = [1] + [0] * 12
        edge = _make_edge(amount_bins, count_bins, num_bins=13)
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=13)
        # active_bin_count reflects REAL activity (1), not the attribute (13).
        assert _close(_scalar(feats, 0, "active_bin_count"), 1.0)

    def test_no_activity_is_zero(self) -> None:
        edge = _make_edge([0.0] * 5, [0] * 5)
        feats = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=8)
        assert _scalar(feats, 0, "active_bin_count") == 0.0
        assert _scalar(feats, 0, "active_bin_fraction") == 0.0


class TestErrorPaths:
    def test_bins_not_a_list_raises(self) -> None:
        edge = _make_edge("22.99;0.0", [1])
        with pytest.raises(EdgeFeatureError):
            _ = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=4)

    def test_missing_attributes_raises(self) -> None:
        with pytest.raises(EdgeFeatureError):
            _ = build_edge_features(
                [{"from_id": "a", "to_id": "b"}], _REFERENCE_EPOCH_S, max_bins=4
            )

    def test_missing_endpoint_raises(self) -> None:
        edge = _make_edge([1.0], [1])
        del edge["from_id"]
        with pytest.raises(EdgeFeatureError):
            _ = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=4)

    def test_invalid_max_bins_raises(self) -> None:
        edge = _make_edge([1.0], [1])
        with pytest.raises(EdgeFeatureError):
            _ = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=0)

    def test_malformed_timestamp_string_raises(self) -> None:
        edge = _make_edge([1.0], [1])
        attrs = edge["attributes"]
        assert isinstance(attrs, dict)
        attrs["first_txn_date"] = "not-a-date"
        with pytest.raises(EdgeFeatureError):
            _ = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=4)


class TestEdgeFeaturesValidation:
    def test_declared_width_must_match_tensor(self) -> None:
        edge = _make_edge([1.0] * 5, [1] * 5)
        good = build_edge_features([edge], _REFERENCE_EPOCH_S, max_bins=5)
        with pytest.raises(EdgeFeatureError):
            _ = EdgeFeatures(
                src_ids=good.src_ids,
                dst_ids=good.dst_ids,
                scalar_feats=good.scalar_feats,
                bin_seq=good.bin_seq,
                max_bins=99,
            )
