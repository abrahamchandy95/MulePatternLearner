"""The nnPU prior grid and its decision (docs/design.md, "Training and selection").

The share of hidden active mules among the unlabelled is unknown, as in a bank, so the prior
is chosen from a grid of multiples of the identified active share measured in the training
world (data.report: identified active mules over scorable accounts, averaged over Mondays):
1, 3 and 10 times it. Config may fix the multiple instead (training.fixed_prior). The prior
is never set from PhantomLedger's identified-share report or from the answer key.

On the development rotation the grid is judged by the selection score alone: when every pair
of multiples has a paired interval of the score difference that contains zero (resampled by
identified mule account, training.selection.paired_interval), no prior is better than another
within noise, and reported runs fix the prior at 1 times the share, which saves two thirds of
their selection training. The decision is recorded with the run; the owner sets
training.fixed_prior from it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from ..config import TrainingConfig
from .selection import Found, paired_interval

# The multiple reported runs fix the prior at when the grid's priors do not differ.
FIXED_MULTIPLE = 1.0


@dataclass(frozen=True)
class Prior:
    """A prior of the grid: its multiple of the identified active share, and its value."""

    multiple: float
    value: float


def grid(share: float, training: TrainingConfig) -> tuple[Prior, ...]:
    """The priors training tries, in the grid's order: the fixed multiple alone when config
    fixes one, otherwise every multiple of prior_multiples; each a prior in (0, 1)."""
    if not 0 < share < 1:
        raise ValueError(f"The identified active share must be in (0, 1), got {share!r}")
    multiples = (
        (training.fixed_prior,) if training.fixed_prior is not None else training.prior_multiples
    )
    found = tuple(Prior(float(multiple), float(multiple) * share) for multiple in multiples)
    for prior in found:
        if not 0 < prior.value < 1:
            raise ValueError(
                f"The prior {prior.multiple:g} times the identified active share {share:g} is "
                f"{prior.value:g}, not in (0, 1)"
            )
    return found


@dataclass(frozen=True)
class Decision:
    """The grid's decision on the development rotation: the interval of the score difference
    of each pair of multiples, the earlier multiple's score minus the later one's."""

    intervals: Mapping[tuple[float, float], tuple[float, float]]

    @property
    def fixed(self) -> bool:
        """Whether every interval contains zero, so reported runs fix the prior at
        FIXED_MULTIPLE times the share."""
        return all(low <= 0 <= high for low, high in self.intervals.values())

    def lines(self) -> list[str]:
        """The decision as a person reads it."""
        found = [
            f"prior {first:g}x against {second:g}x: score difference in [{low:.4f}, {high:.4f}]"
            for (first, second), (low, high) in self.intervals.items()
        ]
        if self.fixed:
            found.append(
                f"No pair of priors differs: reported runs fix the prior at {FIXED_MULTIPLE:g} "
                "times the identified active share"
            )
        else:
            found.append("Some pair of priors differs: reported runs keep the grid")
        return found


def decide(found: Mapping[float, Sequence[Found]]) -> Decision:
    """The decision on the grid from what each multiple's selected model found on the
    selection world, one Found per seed in the same order of seeds for every multiple."""
    if len(found) < 2:
        raise ValueError("A decision on the grid compares at least two priors")
    seeds = {len(runs) for runs in found.values()}
    if len(seeds) != 1 or 0 in seeds:
        raise ValueError("Every prior of a decision needs one selected run per seed")
    return Decision(
        {
            (first, second): paired_interval(found[first], found[second])
            for first, second in combinations(found, 2)
        }
    )
