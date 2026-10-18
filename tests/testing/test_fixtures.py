"""The reference-world fixtures and the transforms the leak tests apply: the catalogue of
look-alikes is one world whose hubs exist, and each transform changes only what it says."""

from __future__ import annotations

import pandas as pd
import pytest

from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.reference.graph import SECOND_MS
from mule_pattern_learner.testing.world import DAY_MS, NO_RING, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS

D = DAY_MS
START = T - LOOKBACK_MS


def test_the_catalogue_is_one_world_whose_hubs_exist_before_the_lookback() -> None:
    graph = fixtures.catalogue().graph()
    for hub in fixtures.HUBS:
        assert graph.accounts[hub].opened_ms < START
    # Every customer joined, and every account opened, before the lookback.
    assert all(joined < START for joined in graph.joined_ms.values())
    assert all(account.opened_ms < START for account in graph.accounts.values())
    mules = {a.id for a in graph.accounts.values() if a.is_mule == 1}
    assert mules == {"a_mule", "a_collector", "a_mule2", "a_mule3"}
    assert [a.id for a in graph.accounts.values() if a.pu_label == 1] == ["a_mule"]


def test_erase_labels_leaves_every_account_no_mule_and_changes_nothing_else() -> None:
    tables = fixtures.catalogue().tables()
    erased = fixtures.erase_labels(tables)
    accounts = erased["Account"]
    for external, labels in ((False, (0, 0, NO_RING, 0, 0)), (True, (-1, -1, NO_RING, 0, 0))):
        rows = accounts[accounts["is_external"] == external]
        assert set(rows[list(fixtures.LABEL_COLUMNS)].itertuples(index=False, name=None)) == {
            labels
        }
    others = [column for column in accounts.columns if column not in fixtures.LABEL_COLUMNS]
    pd.testing.assert_frame_equal(accounts[others], tables["Account"][others])
    assert accounts.dtypes.equals(tables["Account"].dtypes)
    assert all(erased[name] is tables[name] for name in tables if name != "Account")


def test_move_dates_of_record_moves_only_dates_before_the_lookback_and_keeps_them_there() -> None:
    w = WorldBuilder()
    w.customer("old", T - 400 * D, closed_ms=T - 200 * D)
    w.customer("recent", T - 30 * D)
    w.customer("edge", START)
    w.device("p_old", "d_1", T - 300 * D, T - 100 * D)
    w.device("p_recent", "d_1", T - 20 * D)
    tables = w.tables()
    moved = fixtures.move_dates_of_record(tables, T, LOOKBACK_MS)
    for name in tables:
        assert moved[name].dtypes.equals(tables[name].dtypes)
    parties = dict(zip(moved["Party"]["id"], moved["Party"]["joined_ms"], strict=True))
    assert parties["p_old"] < T - 400 * D and parties["p_recent"] == T - 30 * D
    # A date exactly at the lookback's start is not before it, and stays.
    assert parties["p_edge"] == START
    accounts = {
        row[0]: row[1:]
        for row in moved["Account"][["id", "opened_ms", "closed_ms"]].itertuples(
            index=False, name=None
        )
    }
    assert accounts["a_old"][1] < T - 200 * D and accounts["a_recent"] == (T - 30 * D, 0)
    links = moved["Party_Uses_Device"]
    assert list(links["valid_to_ms"]) == [T - 100 * D, 0]
    starts = list(links["valid_from_ms"])
    assert starts[0] < T - 300 * D and starts[1] == T - 20 * D
    # Each moves by its own keyed amount, the same on every run, and never past a year.
    again = fixtures.move_dates_of_record(tables, T, LOOKBACK_MS)
    assert all(again[name].equals(moved[name]) for name in tables)
    shift = T - 400 * D - parties["p_old"]
    assert 0 < shift <= fixtures.MOST_MOVED_MS
    assert shift != T - 400 * D - accounts["a_old"][0]


def test_move_dates_of_record_refuses_to_move_a_date_before_1970_01_02() -> None:
    w = WorldBuilder()
    w.party("p_first", fixtures.EARLIEST_MS + 1)
    with pytest.raises(ValueError, match="before 1970-01-02"):
        fixtures.move_dates_of_record(w.tables(), T, LOOKBACK_MS)


def test_swap_same_second_ids_gives_each_seconds_ids_back_reversed() -> None:
    w = WorldBuilder()
    for name in ("a", "b"):
        w.customer(name, T - 400 * D)
    at = T - 3 * D
    w.pay("x_3", "a_a", "a_b", at, 3.0, "ach", device="d_1")
    w.pay("x_1", "a_a", "a_b", at + 999, 1.0, "card")
    w.pay("z_2", "a_b", "a_a", at + 500, 2.0, "zelle", ip="i_1")
    w.pay("x_alone", "a_a", "a_b", at + SECOND_MS, 4.0, "ach")
    tables = w.tables()
    swapped, original = fixtures.swap_same_second_ids(tables)
    # Of x_1, x_3 and z_2, all of one second, the first by id takes the last's id.
    assert original == {"z_2": "x_1", "x_3": "x_3", "x_1": "z_2", "x_alone": "x_alone"}
    before, renamed = w.graph(), ReferenceGraph(swapped)
    for new, old in original.items():
        payment, was = renamed.payments[new], before.payments[old]
        assert (payment.cents, payment.rail, payment.ts, payment.sender, payment.receiver) == (
            was.cents,
            was.rail,
            was.ts,
            was.sender,
            was.receiver,
        )
        assert (payment.device, payment.ip) == (was.device, was.ip)
    assert swapped["Payment_Transaction"]["id"].dtype == tables["Payment_Transaction"]["id"].dtype
