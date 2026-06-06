from __future__ import annotations

from typing import cast

import pytest
import torch

from mule_pattern_learner.pyg.graph_store import (
    GraphStoreError,
    TigerGraphGraphStore,
)
from mule_pattern_learner.pyg.node_id_mapper import NodeIDMapper
from mule_pattern_learner.tigergraph.client import Client
from torch_geometric.data import EdgeLayout


class _FakeConn:
    pass


class _FakeClient:
    def __init__(self) -> None:
        self.conn: _FakeConn = _FakeConn()

    @property
    def graphname(self) -> str:
        return "Test_Graph"


def _store() -> TigerGraphGraphStore:
    client = cast(Client, cast(object, _FakeClient()))
    mapper = NodeIDMapper()
    return TigerGraphGraphStore(client=client, mapper=mapper)


class TestEdgeAttrs:
    def test_reports_six_edge_types(self) -> None:
        store = _store()
        attrs = store.get_all_edge_attrs()
        assert len(attrs) == 6

    def test_includes_has_paid(self) -> None:
        store = _store()
        attrs = store.get_all_edge_attrs()
        types = {a.edge_type for a in attrs}
        assert ("Account", "HAS_PAID", "Account") in types
        assert ("Party", "Party_Entity", "Resolved_Entity") in types

    def test_all_coo_layout(self) -> None:
        store = _store()
        attrs = store.get_all_edge_attrs()
        assert all(a.layout == EdgeLayout.COO for a in attrs)


class TestReadOnly:
    def test_put_returns_false(self) -> None:
        store = _store()
        ei = (torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long))
        ok = store.put_edge_index(ei, edge_type=("Account", "HAS_PAID", "Account"), layout="coo")
        assert ok is False

    def test_remove_returns_false(self) -> None:
        store = _store()
        ok = store.remove_edge_index(edge_type=("Account", "HAS_PAID", "Account"), layout="coo")
        assert ok is False


class TestUnknownType:
    def test_unknown_edge_type_raises(self) -> None:
        store = _store()
        with pytest.raises(GraphStoreError):
            _ = store.get_edge_index(edge_type=("Account", "NOT_A_REL", "Account"), layout="coo")
