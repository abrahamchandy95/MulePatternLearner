"""Split eligibility, the context keys of a split and the cutoff clocks of its dates."""

from __future__ import annotations

import pandas as pd
import pytest

from mule_pattern_learner.contract.clock import cutoff_ms, timestamp
from mule_pattern_learner.contract.graph_schema import SPLIT_PHASE, ContextKey
from mule_pattern_learner.data.splits import (
    eligible_mask,
    marginal_mask,
    resolve_cutoff,
    resolve_cutoffs,
    sample_keys,
)

ACCOUNTS = pd.DataFrame(
    {
        "account_id": ["a", "b", "c", "d"],
        "split": ["train", "train", "test", "test"],
        "first_seen_ts_ms": [
            timestamp("2024-01-01"),
            timestamp("2024-07-01"),
            timestamp("2024-01-01"),
            timestamp("2024-12-31"),
        ],
        "in_marginal": [True, False, True, True],
    }
)


class Clocks:
    """A CutoffReader whose last visible event is the cutoff's day of the year."""

    def last_visible_seqs(self, cutoff_times: list[int]) -> dict[int, int]:
        return {ms: pd.Timestamp(ms, unit="ms").dayofyear for ms in cutoff_times}


def test_eligible_accounts_existed_before_the_cutoff_date() -> None:
    assert eligible_mask(ACCOUNTS, "train", "2024-07-01").tolist() == [True, False, False, False]
    assert eligible_mask(ACCOUNTS, "test", "2025-01-01").tolist() == [False, False, True, True]
    assert marginal_mask(ACCOUNTS).tolist() == [True, False, True, True]


def test_keys_carry_the_cutoff_the_scope_and_the_split_phase() -> None:
    manifest = {"cutoff_seqs": {"2025-01-01": 30}, "source": {"scope_id": "scope"}}
    keys = sample_keys(ACCOUNTS.iloc[2:], "2025-01-01", manifest)
    phase = SPLIT_PHASE["test"]
    assert keys == [
        ContextKey("Account", name, 30, cutoff_ms("2025-01-01"), "scope", phase)
        for name in ("c", "d")
    ]


def test_cutoffs_are_one_past_the_last_visible_event() -> None:
    # 2024-07-01 minus 1 ms is day 182 of the leap year.
    assert resolve_cutoffs(Clocks(), ["2024-07-01"]) == {"2024-07-01": 183}
    assert resolve_cutoff(Clocks(), "2024-07-01") == (183, cutoff_ms("2024-07-01"))

    class Empty:
        def last_visible_seqs(self, cutoff_times: list[int]) -> dict[int, int]:
            return dict.fromkeys(cutoff_times, 0)

    with pytest.raises(ValueError, match="visible before 2024-07-01"):
        resolve_cutoffs(Empty(), ["2024-07-01"])
