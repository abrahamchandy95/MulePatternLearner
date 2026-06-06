from __future__ import annotations

import math
from typing import cast

import pytest

from mule_pattern_learner.schema.node_features import (
    NUM_ACCOUNT_FEATURES,
    NodeFeatureError,
    NodeFeatures,
    Transform,
    account_feature_names,
    build_account_features,
    log1p_compress,
    symlog_compress,
)
from mule_pattern_learner.schema.node_features import (
    _ACCOUNT_FEATURE_TRANSFORMS as ACCOUNT_FEATURE_TRANSFORMS,  # pyright: ignore[reportPrivateUsage]
)


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)


def _account_vertex(overrides: dict[str, object], node_id: str = "A1") -> dict[str, object]:
    attrs: dict[str, object] = {name: 0.0 for name in account_feature_names()}
    for k, v in overrides.items():
        attrs[k] = v
    return {"v_id": node_id, "v_type": "Account", "attributes": attrs}


def _feat(features: NodeFeatures, row_index: int, name: str) -> float:
    rows = cast(list[list[object]], features.feats.tolist())
    cell = rows[row_index][account_feature_names().index(name)]
    assert isinstance(cell, (int, float))
    return float(cell)


class TestTransforms:
    def test_log1p_matches(self) -> None:
        assert _close(log1p_compress(1000.0), math.log1p(1000.0))

    def test_log1p_clamps_negative(self) -> None:
        assert log1p_compress(-5.0) == 0.0

    def test_symlog_non_negative_equals_log1p(self) -> None:
        for v in (0.0, 5.0, 1000.0):
            assert _close(symlog_compress(v), log1p_compress(v))

    def test_symlog_preserves_sign(self) -> None:
        assert _close(symlog_compress(-500.0), -math.log1p(500.0))

    def test_symlog_zero(self) -> None:
        assert symlog_compress(0.0) == 0.0


class TestTransformPolicy:
    def test_every_feature_has_a_transform(self) -> None:
        assert set(ACCOUNT_FEATURE_TRANSFORMS) == set(account_feature_names())

    def test_net_flow_is_symlog(self) -> None:
        assert ACCOUNT_FEATURE_TRANSFORMS["net_flow"] is Transform.SYMLOG

    def test_amounts_are_log1p(self) -> None:
        assert ACCOUNT_FEATURE_TRANSFORMS["out_amount"] is Transform.LOG1P
        assert ACCOUNT_FEATURE_TRANSFORMS["in_amount"] is Transform.LOG1P

    def test_ratios_are_identity(self) -> None:
        assert ACCOUNT_FEATURE_TRANSFORMS["clustering_coef"] is Transform.IDENTITY

    def test_flag_is_boolean(self) -> None:
        assert ACCOUNT_FEATURE_TRANSFORMS["is_external"] is Transform.BOOLEAN


class TestShapes:
    def test_shape(self) -> None:
        v = _account_vertex({})
        feats = build_account_features([v, v, v])
        assert feats.num_nodes == 3
        assert tuple(feats.feats.shape) == (3, NUM_ACCOUNT_FEATURES)
        assert feats.feature_names == account_feature_names()

    def test_node_ids_preserved(self) -> None:
        feats = build_account_features([_account_vertex({}, "ACC_X"), _account_vertex({}, "ACC_Y")])
        assert feats.node_ids == ("ACC_X", "ACC_Y")


class TestValues:
    def test_log1p_family(self) -> None:
        feats = build_account_features([_account_vertex({"out_amount": 1000.0})])
        assert _close(_feat(feats, 0, "out_amount"), math.log1p(1000.0))

    def test_symlog_family_negative(self) -> None:
        feats = build_account_features([_account_vertex({"net_flow": -500.0})])
        assert _close(_feat(feats, 0, "net_flow"), -math.log1p(500.0))

    def test_identity_family(self) -> None:
        feats = build_account_features(
            [_account_vertex({"clustering_coef": 0.37, "pass_through_ratio": 0.5})]
        )
        assert _close(_feat(feats, 0, "clustering_coef"), 0.37)
        assert _close(_feat(feats, 0, "pass_through_ratio"), 0.5)

    def test_boolean_family(self) -> None:
        feats = build_account_features([_account_vertex({"is_external": 1})])
        assert _feat(feats, 0, "is_external") == 1.0

    def test_boolean_from_python_bool(self) -> None:
        feats = build_account_features([_account_vertex({"is_external": True})])
        assert _feat(feats, 0, "is_external") == 1.0

    def test_missing_attribute_defaults_zero(self) -> None:
        vertex: dict[str, object] = {
            "v_id": "A1",
            "v_type": "Account",
            "attributes": {"out_amount": 100.0},
        }
        feats = build_account_features([vertex])
        assert _close(_feat(feats, 0, "out_amount"), math.log1p(100.0))
        assert _feat(feats, 0, "in_amount") == 0.0


class TestClampSentinel:
    def test_sentinel_minus_one_maps_to_zero(self) -> None:
        feats = build_account_features(
            [
                _account_vertex(
                    {
                        "days_since_last_txn": -1.0,
                        "account_age_days": -1.0,
                        "mean_inter_txn_days": -1.0,
                    }
                )
            ]
        )
        assert _feat(feats, 0, "days_since_last_txn") == 0.0
        assert _feat(feats, 0, "account_age_days") == 0.0
        assert _feat(feats, 0, "mean_inter_txn_days") == 0.0

    def test_real_value_passes_through(self) -> None:
        feats = build_account_features(
            [_account_vertex({"days_since_last_txn": 36.97, "account_age_days": 181.24})]
        )
        assert _close(_feat(feats, 0, "days_since_last_txn"), 36.97)
        assert _close(_feat(feats, 0, "account_age_days"), 181.24)

    def test_zero_is_not_clamped(self) -> None:
        # A real 0 must stay 0 (only the -1 sentinel is remapped).
        feats = build_account_features([_account_vertex({"days_since_last_txn": 0.0})])
        assert _feat(feats, 0, "days_since_last_txn") == 0.0

    def test_policy_assignment(self) -> None:
        for name in (
            "days_since_last_txn",
            "account_age_days",
            "mean_inter_txn_days",
        ):
            assert ACCOUNT_FEATURE_TRANSFORMS[name] is Transform.CLAMP_SENTINEL


class TestErrorPaths:
    def test_missing_v_id_raises(self) -> None:
        bad: dict[str, object] = {"v_type": "Account", "attributes": {}}
        with pytest.raises(NodeFeatureError):
            _ = build_account_features([bad])

    def test_missing_attributes_raises(self) -> None:
        bad: dict[str, object] = {"v_id": "A1"}
        with pytest.raises(NodeFeatureError):
            _ = build_account_features([bad])

    def test_non_numeric_attribute_raises(self) -> None:
        feats_vertex = _account_vertex({"out_amount": "not-a-number"})
        with pytest.raises(NodeFeatureError):
            _ = build_account_features([feats_vertex])

    def test_vertex_not_dict_raises(self) -> None:
        with pytest.raises(NodeFeatureError):
            _ = build_account_features(["not-a-vertex"])


class TestNodeFeaturesValidation:
    def test_shape_mismatch_raises(self) -> None:
        good = build_account_features([_account_vertex({})])
        with pytest.raises(NodeFeatureError):
            _ = NodeFeatures(
                node_ids=("A1", "A2"),
                feats=good.feats,
                feature_names=account_feature_names(),
            )
