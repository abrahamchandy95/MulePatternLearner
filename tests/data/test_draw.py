"""The keyed draws: the hash of an (account, Monday), pinned; the training draw, uniform and
nesting; the fallback sample, at its inclusion chance (docs/design.md, "Training world")."""

from __future__ import annotations

import hashlib

import pytest
from scipy.stats import chisquare

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.data import draw
from mule_pattern_learner.data.draw import SCALE, draw_hash, lowest, sampled

KEY = DEFAULT_CONFIG.data.draw_key
T = 1_719_792_000_000  # 2024-07-01 00:00 UTC, a Monday
WEEK = 7 * 86_400_000
POOL = [f"a_n{index:02d}" for index in range(12)]


def test_the_hashes_are_pinned() -> None:
    # Golden: blake2b keyed by b"mule-draw-v1" and four zero bytes, 8 bytes long, of
    # "<account id>|<Monday ms>", big-endian. A change here changes every draw and sample.
    assert KEY == b"mule-draw-v1\x00\x00\x00\x00"
    assert draw_hash(KEY, "a_n00", T) == 0x9C1A095EEF539F33
    assert draw_hash(KEY, "a_n00", T - 13 * WEEK) == 0xC5951F7F9200E281
    assert draw_hash(KEY, "acct-0001", T) == 0x6C3EFA4183FAE4C5
    assert draw_hash(KEY, "a_n10", T) == 0x0311A7E6086A4266
    digest = hashlib.blake2b(b"a_n00|1719792000000", key=KEY, digest_size=8).digest()
    assert draw_hash(KEY, "a_n00", T) == int.from_bytes(digest, "big")
    # Another key or another Monday gives another hash.
    assert draw_hash(b"other", "a_n00", T) != draw_hash(KEY, "a_n00", T)


def test_the_draw_takes_the_lowest_hashes() -> None:
    # Of the twelve, a_n10 (0x0311...), a_n04 (0x0af2...) and a_n02 (0x47e8...) hash lowest.
    assert lowest(KEY, POOL, T, 3) == ["a_n10", "a_n04", "a_n02"]
    assert lowest(KEY, reversed(POOL), T, 3) == ["a_n10", "a_n04", "a_n02"]
    assert lowest(KEY, POOL, T, 0) == []
    assert sorted(lowest(KEY, POOL, T, 50)) == POOL
    with pytest.raises(ValueError, match="not negative"):
        lowest(KEY, POOL, T, -1)


def test_hashes_that_tie_are_broken_by_account_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 64-bit hash practically never ties, so the rule is shown with a hash that always does:
    # the lowest two of b, a and c are a and b, whatever order they come in.

    def tied(key: bytes, account_id: str, t_ms: int) -> int:
        return 7

    monkeypatch.setattr(draw, "draw_hash", tied)
    assert lowest(KEY, ["b", "a", "c"], T, 2) == ["a", "b"]
    assert lowest(KEY, ["c", "b", "a"], T, 2) == ["a", "b"]
    assert lowest(KEY, ["c", "b", "a"], T, 3) == ["a", "b", "c"]


def test_the_draw_is_uniform() -> None:
    # 10,000 draws of 5 from a pool of 50, one per Monday: each account is drawn 1,000 times in
    # expectation. A chi-square test of the counts at the 0.1% level.
    pool = [f"acct-{index:03d}" for index in range(50)]
    counts = dict.fromkeys(pool, 0)
    for week in range(10_000):
        for account in lowest(KEY, pool, T + week * WEEK, 5):
            counts[account] += 1
    assert sum(counts.values()) == 50_000
    assert chisquare(list(counts.values())).pvalue > 0.001


def test_the_draw_nests_when_accounts_join_or_leave_the_pool() -> None:
    pool = [f"acct-{index:04d}" for index in range(2_000)]
    drawn = lowest(KEY, pool, T, 500)
    joined = [f"new-{index}" for index in range(5)]
    grown = lowest(KEY, pool + joined, T, 500)
    # Every account the larger pool draws anew is one that joined, so at most 5 of the draw
    # change and the rest keep their cached rows.
    assert set(grown) - set(drawn) <= set(joined)
    assert len(set(drawn) & set(grown)) >= 495
    left = drawn[:5]
    shrunk = lowest(KEY, [account for account in pool if account not in left], T, 500)
    assert set(drawn) - set(shrunk) == set(left)
    assert len(set(shrunk) - set(drawn)) == 5


def test_the_sample_includes_each_account_with_chance_one_over_its_weight() -> None:
    assert sampled(KEY, POOL, T, 10) == [
        account for account in POOL if draw_hash(KEY, account, T) < SCALE // 10
    ]
    # a_n04 and a_n10 hash below 2^64 / 10 (their shares 0.043 and 0.012).
    assert sampled(KEY, POOL, T, 10) == ["a_n04", "a_n10"]
    assert sampled(KEY, POOL, T, 1) == POOL
    many = [f"acct-{index:05d}" for index in range(100_000)]
    # 9,907 of 100,000: within three standard deviations (about 95) of 10,000.
    assert abs(len(sampled(KEY, many, T, 10)) - 10_000) < 300
    with pytest.raises(ValueError, match="at least 1"):
        sampled(KEY, POOL, T, 0)


def test_the_draw_lies_inside_the_sample_when_it_is_smaller() -> None:
    # The draw and the sample read one hash, so a world that trains in one rotation and selects
    # in another caches its drawn accounts for both.
    many = [f"acct-{index:05d}" for index in range(10_000)]
    assert set(lowest(KEY, many, T, 500)) <= set(sampled(KEY, many, T, 10))
