from __future__ import annotations

from typing import cast

import pytest
import torch

from mule_pattern_learner.pyg.feature_store import (
    FeatureStoreError,
    TigerGraphFeatureStore,
)
from mule_pattern_learner.pyg.node_id_mapper import NodeIDMapper
from mule_pattern_learner.schema.node_features import NUM_ACCOUNT_FEATURES
from mule_pattern_learner.tigergraph.client import Client


class _FakeConn:
    pass


class _FakeClient:
    def __init__(self) -> None:
        self.conn: _FakeConn = _FakeConn()

    @property
    def graphname(self) -> str:
        return "Test_Graph"


def _store() -> TigerGraphFeatureStore:
    client = cast(Client, cast(object, _FakeClient()))
    mapper = NodeIDMapper()
    return TigerGraphFeatureStore(client=client, mapper=mapper)


class TestTensorSize:
    def test_account_size_is_feature_dim(self) -> None:
        store = _store()
        size = store.get_tensor_size(group_name="Account", attr_name="x")
        assert size == (NUM_ACCOUNT_FEATURES,)

    def test_unknown_type_size_is_none(self) -> None:
        store = _store()
        size = store.get_tensor_size(group_name="Party", attr_name="x")
        assert size is None


class TestTensorAttrs:
    def test_lists_account_x(self) -> None:
        store = _store()
        attrs = store.get_all_tensor_attrs()
        names = {(a.group_name, a.attr_name) for a in attrs}
        assert ("Account", "x") in names


class TestReadOnly:
    def test_put_returns_false(self) -> None:
        store = _store()
        ok = store.put_tensor(
            torch.tensor([0.0]),
            group_name="Account",
            attr_name="x",
            index=torch.tensor([0], dtype=torch.long),
        )
        assert ok is False

    def test_remove_returns_false(self) -> None:
        store = _store()
        ok = store.remove_tensor(
            group_name="Account",
            attr_name="x",
            index=torch.tensor([0], dtype=torch.long),
        )
        assert ok is False


class TestValidation:
    def test_wrong_attr_name_raises(self) -> None:
        store = _store()
        with pytest.raises(FeatureStoreError):
            _ = store.get_tensor(
                group_name="Account",
                attr_name="pos",
                index=torch.tensor([0], dtype=torch.long),
            )

    def test_unknown_type_raises(self) -> None:
        store = _store()
        with pytest.raises(FeatureStoreError):
            _ = store.get_tensor(
                group_name="Party",
                attr_name="x",
                index=torch.tensor([0], dtype=torch.long),
            )
