from __future__ import annotations

from typing import cast

import pytest

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.derivation import (
    GraphTemporalSpec,
    TemporalDerivationError,
    derive_temporal_spec,
)
from mule_pattern_learner.tigergraph.temporal import flat_edge_dim


class _FakeConn:
    _result: list[object]
    calls: list[tuple[str, object]]

    def __init__(self, result: list[object]) -> None:
        self._result = result
        self.calls = []

    def runInstalledQuery(self, name: str, params: object = None) -> list[object]:
        self.calls.append((name, params))
        return self._result


class _FakeClient:
    conn: _FakeConn

    def __init__(self, result: list[object]) -> None:
        self.conn = _FakeConn(result)


def _as_client(fake: _FakeClient) -> Client:
    return cast(Client, cast(object, fake))


def _consistent(n: int) -> list[object]:
    return [
        {"max_bins": n},
        {"max_amount_len": n},
        {"max_count_len": n},
    ]


class TestGraphTemporalSpec:
    def test_edge_dim_matches_flat_edge_dim(self) -> None:
        spec = GraphTemporalSpec(max_bins=13)
        assert spec.edge_dim == flat_edge_dim(13)

    def test_edge_dim_tracks_max_bins(self) -> None:
        assert GraphTemporalSpec(max_bins=26).edge_dim == flat_edge_dim(26)

    def test_rejects_zero(self) -> None:
        with pytest.raises(TemporalDerivationError):
            _ = GraphTemporalSpec(max_bins=0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(TemporalDerivationError):
            _ = GraphTemporalSpec(max_bins=-3)


class TestDeriveTemporalSpec:
    def test_returns_spec_with_derived_width(self) -> None:
        spec = derive_temporal_spec(_as_client(_FakeClient(_consistent(13))))
        assert spec.max_bins == 13
        assert spec.edge_dim == flat_edge_dim(13)

    def test_calls_the_derive_query(self) -> None:
        fake = _FakeClient(_consistent(13))
        _ = derive_temporal_spec(_as_client(fake))
        assert fake.conn.calls == [("derive_max_bins", {})]

    def test_raises_on_inconsistent_counts(self) -> None:
        bad: list[object] = [
            {"max_bins": 13},
            {"max_amount_len": 13},
            {"max_count_len": 12},
        ]
        with pytest.raises(TemporalDerivationError):
            _ = derive_temporal_spec(_as_client(_FakeClient(bad)))

    def test_raises_on_missing_key(self) -> None:
        with pytest.raises(TemporalDerivationError):
            _ = derive_temporal_spec(_as_client(_FakeClient([{"max_bins": 13}])))

    def test_raises_on_non_int(self) -> None:
        bad: list[object] = [
            {"max_bins": "13"},
            {"max_amount_len": 13},
            {"max_count_len": 13},
        ]
        with pytest.raises(TemporalDerivationError):
            _ = derive_temporal_spec(_as_client(_FakeClient(bad)))
