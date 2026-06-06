from __future__ import annotations

from typing import cast

from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.pyg.fanout import NeighborFanout
from mule_pattern_learner.pyg.node_id_mapper import NodeIDMapper
from mule_pattern_learner.tigergraph.client import Client


class _FakeConn:
    """Minimal stand-in; the backend never calls it in these tests."""

    pass


class _FakeClient:
    """Fake Client exposing the attributes the backend reads."""

    def __init__(self) -> None:
        self.conn: _FakeConn = _FakeConn()

    @property
    def graphname(self) -> str:
        return "Test_Graph"


def _backend() -> TigerGraphRemoteBackend:
    client = cast(Client, cast(object, _FakeClient()))
    return TigerGraphRemoteBackend(client)


class TestBackendIdentity:
    def test_holds_client(self) -> None:
        backend = _backend()
        assert backend.client is not None

    def test_mapper_is_node_id_mapper(self) -> None:
        backend = _backend()
        assert isinstance(backend.mapper, NodeIDMapper)

    def test_same_mapper_returned_each_access(self) -> None:
        backend = _backend()
        assert backend.mapper is backend.mapper


class TestSamplerSharesMapper:
    def test_sampler_uses_backend_mapper(self) -> None:
        # The crucial property: the sampler the backend builds shares the
        # backend's exact mapper object, so integers it assigns are reversible
        # by a feature store built from the same backend.
        backend = _backend()
        sampler = backend.make_sampler(seed_ids=("A1", "A2"))
        # access the sampler's mapper via its private attribute for the test
        sampler_mapper = cast(NodeIDMapper, getattr(sampler, "_mapper"))
        assert sampler_mapper is backend.mapper

    def test_registration_through_sampler_visible_on_backend(self) -> None:
        backend = _backend()
        sampler = backend.make_sampler(seed_ids=("A1",))
        sampler_mapper = cast(NodeIDMapper, getattr(sampler, "_mapper"))
        # register via the sampler's mapper; backend's mapper must see it
        ints = sampler_mapper.register("Account", ["A0000000001", "A0000000002"])
        assert ints == [0, 1]
        assert backend.mapper.num_nodes("Account") == 2
        assert backend.mapper.to_string("Account", 0) == "A0000000001"

    def test_two_samplers_share_one_mapper(self) -> None:
        # Two samplers from the same backend share the same persistent mapper,
        # so ids seen by one are known to the other (consistent across the run).
        backend = _backend()
        s1 = backend.make_sampler(seed_ids=("A1",))
        s2 = backend.make_sampler(seed_ids=("A2",))
        m1 = cast(NodeIDMapper, getattr(s1, "_mapper"))
        m2 = cast(NodeIDMapper, getattr(s2, "_mapper"))
        assert m1 is m2 is backend.mapper


class TestSamplerConfig:
    def test_default_fanout(self) -> None:
        backend = _backend()
        sampler = backend.make_sampler(seed_ids=("A1",))
        fanout = cast(NeighborFanout, getattr(sampler, "_fanout"))
        assert fanout.has_paid == 15

    def test_custom_fanout_passed_through(self) -> None:
        backend = _backend()
        custom = NeighborFanout(has_paid=25)
        sampler = backend.make_sampler(seed_ids=("A1",), fanout=custom)
        fanout = cast(NeighborFanout, getattr(sampler, "_fanout"))
        assert fanout.has_paid == 25

    def test_seed_ids_stored(self) -> None:
        backend = _backend()
        sampler = backend.make_sampler(seed_ids=("A1", "A2", "A3"))
        seeds = cast("tuple[str, ...]", getattr(sampler, "_seed_ids"))
        assert seeds == ("A1", "A2", "A3")
