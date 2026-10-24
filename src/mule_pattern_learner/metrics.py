"""The Monday rankings of the test world (docs/design.md, "Rankings" and "Metrics").

Pure numpy: arrays in, numbers out. MondayRanking orders each Monday's account-Mondays by logit,
highest first, in blocks of tied logits, and gives each the share of it that each budget, the top
1% or 5% of its Monday's ranked weight, finds: a budget that ends inside a block finds the same
share of each of its accounts, the expected result of ordering them at random. From those shares:
recall and precision per Monday and pooled over Mondays (budget_found), the chance that an
account is flagged on any of its Mondays (ever_flagged), and the expected days to its first flag
(first_flag). budget_name names a budget in a metric.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def budget_name(fraction: float) -> str:
    """How the metrics of a review budget name it: 0.05 is "5pct" (precision_at_5pct)."""
    return f"{round(fraction * 100)}pct"


# The budgets of the test rankings: the top 1% and 5% of each Monday's ranked weight.
TEST_BUDGETS = (0.01, 0.05)


def _blocks_of(
    monday: NDArray[np.int64], score: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.intp]]:
    """The order of rows by Monday, then score highest first; where each Monday and each block
    of tied scores starts in that order."""
    order = np.lexsort((-score, monday)).astype(np.intp)
    mondays, scores = monday[order], score[order]
    new_monday = np.r_[True, mondays[1:] != mondays[:-1]] if len(order) else np.zeros(0, bool)
    new_block = new_monday | (np.r_[True, scores[1:] != scores[:-1]] if len(order) else new_monday)
    return order, new_monday, new_block, np.cumsum(new_block) - 1


class MondayRanking:
    """The account-Mondays of one ranking: each Monday's ordered by score, highest first, in
    blocks of tied scores. Scores are compared within a Monday only."""

    def __init__(self, monday: NDArray[np.int64], score: NDArray[np.floating]) -> None:
        values = np.asarray(score, dtype=np.float64)
        mondays = np.asarray(monday, dtype=np.int64)
        if values.shape != mondays.shape or values.ndim != 1:
            raise ValueError(f"{values.shape} scores for {mondays.shape} Mondays")
        if not np.isfinite(values).all():
            raise ValueError("A ranked score is not finite")
        order, new_monday, new_block, block = _blocks_of(mondays, values)
        self.size = len(values)
        self.block = np.empty(self.size, dtype=np.intp)
        self.block[order] = block
        self.blocks = int(new_block.sum())
        monday_index = np.cumsum(new_monday) - 1
        self.monday_of_row = np.empty(self.size, dtype=np.intp)
        self.monday_of_row[order] = monday_index
        # Each block's Monday and score, and each Monday's value and first block.
        self.block_monday = monday_index[new_block]
        self.block_score = values[order][new_block]
        self.mondays = mondays[order][new_monday]
        self.first_block = block[new_monday]

    def _weights(
        self, weight: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Each block's weight, the weight ranked above it in its Monday, and each Monday's
        weight."""
        if weight.shape != (self.size,):
            raise ValueError(f"{weight.shape} weights for {self.size} ranked rows")
        block = np.bincount(self.block, weights=weight, minlength=self.blocks).astype(np.float64)
        before = np.cumsum(block) - block
        above = before - before[self.first_block][self.block_monday]
        monday = np.bincount(self.block_monday, weights=block, minlength=len(self.mondays))
        return block, above, monday.astype(np.float64)

    @staticmethod
    def _share(
        size: NDArray[np.float64], above: NDArray[np.float64], tied: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The share of a block of weight tied, with above ranked over it, that a budget of
        size finds; a block of no weight is found whole when the budget reaches past what is
        above it."""
        found = np.divide(size - above, tied, out=np.zeros_like(size), where=tied > 0)
        return np.where(tied > 0, np.clip(found, 0.0, 1.0), (above < size).astype(np.float64))

    def shares(
        self, weight: NDArray[np.float64], budgets: Sequence[float] = TEST_BUDGETS
    ) -> NDArray[np.float64]:
        """The share of each row each budget finds: a row per row, a column per budget."""
        block, above, monday = self._weights(weight)
        found = np.stack(
            [self._share(b * monday[self.block_monday], above, block) for b in budgets], axis=1
        )
        return found[self.block]

    def rank_fraction(self, weight: NDArray[np.float64]) -> NDArray[np.float64]:
        """Each row's place in its Monday as a share of the Monday's weight: the weight ranked
        above its block and half its block's."""
        block, above, monday = self._weights(weight)
        total = monday[self.block_monday]
        place = np.divide(above + block / 2, total, out=np.zeros_like(total), where=total > 0)
        return place[self.block]

    def against(
        self,
        weight: NDArray[np.float64],
        monday: NDArray[np.int64],
        score: NDArray[np.floating],
        budgets: Sequence[float] = TEST_BUDGETS,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """For rows outside the ranking, each on a Monday of it, the share each budget of its
        Monday finds and its rank fraction, against the ranked rows alone, as if each were
        added with no weight."""
        block, _, totals = self._weights(weight)
        values = np.asarray(score, dtype=np.float64)
        shares = np.zeros((len(values), len(budgets)))
        fractions = np.zeros(len(values))
        index = {int(value): position for position, value in enumerate(self.mondays.tolist())}
        ends = np.r_[self.first_block[1:], self.blocks]
        for row, (day, value) in enumerate(zip(monday.tolist(), values.tolist(), strict=True)):
            if int(day) not in index:
                raise ValueError(f"No ranked row is on the Monday {day}")
            position = index[int(day)]
            first, last = int(self.first_block[position]), int(ends[position])
            scores = self.block_score[first:last]
            # The blocks of the Monday run highest first: those scoring above the row, then
            # the block it ties with, if any.
            higher = int(np.searchsorted(-scores, -value, side="left"))
            ties = higher < len(scores) and scores[higher] == value
            over = float(block[first : first + higher].sum())
            tied = float(block[first + higher]) if ties else 0.0
            total = float(totals[position])
            for column, budget in enumerate(budgets):
                shares[row, column] = self._share(
                    np.array([budget * total]), np.array([over]), np.array([tied])
                )[0]
            fractions[row] = (over + tied / 2) / total if total > 0 else 0.0
        return shares, fractions

    def budget_found(
        self,
        shares: NDArray[np.float64],
        positive: NDArray[np.bool_],
        weight: NDArray[np.float64],
        budgets: Sequence[float] = TEST_BUDGETS,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Per Monday (in increasing order): the positive weight each budget found (a column
        per budget), the positive weight, and each budget's size in weight."""
        _, _, totals = self._weights(weight)
        rows = self.monday_of_row
        found = np.stack(
            [
                np.bincount(
                    rows,
                    weights=np.where(positive, weight * shares[:, column], 0.0),
                    minlength=len(self.mondays),
                )
                for column in range(len(budgets))
            ],
            axis=1,
        )
        positives = np.bincount(
            rows, weights=np.where(positive, weight, 0.0), minlength=len(self.mondays)
        )
        sizes = np.outer(totals, np.asarray(budgets, dtype=np.float64))
        return found.astype(np.float64), positives.astype(np.float64), sizes


def ever_flagged(
    shares: NDArray[np.float64], group: NDArray[np.intp], groups: int
) -> NDArray[np.float64]:
    """The chance each group (an account) is flagged on at least one of its rows, each row
    flagged with its share apart from the others: 1 minus the product of 1 minus each
    share."""
    with np.errstate(divide="ignore"):
        missed = np.log1p(-np.clip(shares, 0.0, 1.0))
    total = np.zeros(groups)
    np.add.at(total, group, missed)
    return 1.0 - np.exp(total)


def first_flag(days: NDArray[np.float64], shares: NDArray[np.float64]) -> tuple[float, float]:
    """Of one account's rows, in increasing days, each flagged with its share apart from the
    others: the chance it is flagged on any, and the expected days to its first flag given
    that it is (NaN when it never is)."""
    order = np.argsort(days, kind="stable")
    ordered, chances = days[order], np.clip(shares[order], 0.0, 1.0)
    unflagged = np.r_[1.0, np.cumprod(1.0 - chances)[:-1]]
    first = chances * unflagged
    ever = float(first.sum())
    return ever, float(first @ ordered) / ever if ever > 0 else float("nan")
