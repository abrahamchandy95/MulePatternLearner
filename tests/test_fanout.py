from __future__ import annotations

import pytest

from mule_pattern_learner.pyg.fanout import NeighborFanout, NeighborFanoutError


class TestDefaults:
    def test_defaults_construct(self) -> None:
        fanout = NeighborFanout()
        assert fanout.has_paid == 15
        assert fanout.party_acct_round2 == 5

    def test_round2_not_larger_than_round1(self) -> None:
        # The decreasing-fanout invariant: each round-2 count <= its round-1 count.
        fanout = NeighborFanout()
        assert fanout.has_paid_round2 <= fanout.has_paid
        assert fanout.account_account_round2 <= fanout.account_account
        assert fanout.acct_party_round2 <= fanout.acct_party
        assert fanout.party_entity_round2 <= fanout.party_entity
        assert fanout.entity_party_round2 <= fanout.entity_party
        assert fanout.party_acct_round2 <= fanout.party_acct


class TestQueryParams:
    def test_as_query_params_has_all_twelve(self) -> None:
        params = NeighborFanout().as_query_params()
        assert len(params) == 12
        assert set(params) == {
            "fanout_has_paid",
            "fanout_account_account",
            "fanout_acct_party",
            "fanout_party_entity",
            "fanout_entity_party",
            "fanout_party_acct",
            "fanout_has_paid_2",
            "fanout_account_account_2",
            "fanout_acct_party_2",
            "fanout_party_entity_2",
            "fanout_entity_party_2",
            "fanout_party_acct_2",
        }

    def test_query_param_names_map_to_gsql(self) -> None:
        # de-smurfed field -> GSQL parameter name mapping is correct
        params = NeighborFanout(has_paid=20, has_paid_round2=8).as_query_params()
        assert params["fanout_has_paid"] == 20
        assert params["fanout_has_paid_2"] == 8

    def test_as_query_params_all_ints(self) -> None:
        params = NeighborFanout().as_query_params()
        assert all(isinstance(v, int) for v in params.values())

    def test_custom_values_round_trip(self) -> None:
        fanout = NeighborFanout(has_paid=7, party_acct_round2=2)
        params = fanout.as_query_params()
        assert params["fanout_has_paid"] == 7
        assert params["fanout_party_acct_2"] == 2


class TestValidation:
    def test_zero_rejected(self) -> None:
        with pytest.raises(NeighborFanoutError):
            _ = NeighborFanout(has_paid=0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(NeighborFanoutError):
            _ = NeighborFanout(account_account=-1)

    def test_bool_rejected(self) -> None:
        with pytest.raises(NeighborFanoutError):
            _ = NeighborFanout(has_paid=True)

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        fanout = NeighborFanout()
        with pytest.raises(FrozenInstanceError):
            setattr(fanout, "has_paid", 99)
