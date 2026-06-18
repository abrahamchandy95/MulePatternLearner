from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
import random


class Split(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """
    Parameters for a party-grouped train/val/test split.

    val_fraction / test_fraction:
        Proportions of the freely-assignable parties (those not forced to a
        split). The remainder is train. Forced-test accounts are pinned on top,
        so the effective test fraction may exceed test_fraction.
    seed:
        RNG seed; the assignment is deterministic given inputs and seed.
    """

    val_fraction: float = 0.20
    test_fraction: float = 0.15
    seed: int = 1337

    def validate(self) -> None:
        if self.val_fraction < 0.0 or self.test_fraction < 0.0:
            raise ValueError("split fractions must be non-negative.")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1.")


@dataclass(frozen=True, slots=True)
class SplitRecord:
    """Minimal per-account input for splitting: identity and grouping key."""

    account_id: str
    party_id: str | None


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Per-account split assignment, aligned to the input order."""

    account_ids: tuple[str, ...]
    split: tuple[int, ...]

    def summary(self) -> dict[str, int]:
        return {
            "accounts": len(self.account_ids),
            "train": sum(1 for s in self.split if s == Split.TRAIN),
            "val": sum(1 for s in self.split if s == Split.VAL),
            "test": sum(1 for s in self.split if s == Split.TEST),
        }


def _group_accounts_by_party(
    records: Sequence[SplitRecord],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        key = r.party_id if r.party_id is not None else f"__solo__{r.account_id}"
        groups.setdefault(key, []).append(i)
    return groups


def _assign_by_fraction(
    keys: list[str], val_fraction: float, test_fraction: float
) -> tuple[set[str], set[str]]:
    # Slice an (already-shuffled) key list into val / test sets by fraction;
    # the remainder is train. Shared by the plain-group and positive-group paths.
    k = len(keys)
    n_val = int(round(val_fraction * k))
    n_test = int(round(test_fraction * k))
    return set(keys[:n_val]), set(keys[n_val : n_val + n_test])


def _stratify_positive_groups(
    keys: list[str], val_fraction: float, test_fraction: float
) -> tuple[set[str], set[str]]:
    # Distribute the FEW groups that contain a revealed positive across the
    # splits, with a floor of one group each for val and test whenever there are
    # at least three such groups. Proportional slicing alone would round the
    # tiny val/test shares down to zero (the bug that left val with no positives
    # to rank for Proxy-AUC model selection); the floor guarantees coverage.
    k = len(keys)
    if k == 0:
        return set(), set()
    if k == 1:
        # the single positive group is most valuable for training the rare class
        return set(), set()
    if k == 2:
        # one to val (model selection needs it), one to train
        return {keys[0]}, set()
    n_val = max(1, int(round(val_fraction * k)))
    n_test = max(1, int(round(test_fraction * k)))
    # never claim more than available; train keeps the remainder
    if n_val + n_test >= k:
        n_test = max(1, k - n_val - 1)
    return set(keys[:n_val]), set(keys[n_val : n_val + n_test])


def split_accounts(
    records: Sequence[SplitRecord],
    config: SplitConfig,
    force_test: frozenset[str] | None = None,
    stratify: frozenset[str] | None = None,
) -> SplitResult:
    """Assign each account to TRAIN, VAL, or TEST, grouped by party.

    Accounts are grouped by party so that all accounts owned by one party land
    in the same split (no owner straddles the train/test boundary, which would
    leak). force_test names accounts that must be in TEST regardless; any party
    containing a forced account is pinned entirely to TEST.

    stratify names accounts (the revealed positives, pu_label==1) whose party
    groups must be spread across train/val/test rather than left to chance. With
    only a handful of revealed positives, proportional slicing can round val's
    share to zero, leaving Proxy-AUC model selection with no positive to rank.
    Positive-bearing groups are therefore assigned first, with a floor of one
    group each for val and test (when at least three exist). The remaining
    (plain) groups are split by val_fraction / test_fraction, remainder to train.
    """
    config.validate()
    forced: frozenset[str] = force_test if force_test is not None else frozenset()
    strat: frozenset[str] = stratify if stratify is not None else frozenset()
    rng = random.Random(config.seed)

    n = len(records)
    split: list[int] = [-1] * n

    groups = _group_accounts_by_party(records)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    # Partition the non-forced ("free") groups into those that contain a
    # revealed positive (stratified first, for guaranteed coverage) and the rest.
    positive_groups: list[str] = []
    plain_groups: list[str] = []
    for key in group_keys:
        idxs = groups[key]
        if any(records[i].account_id in forced for i in idxs):
            for i in idxs:
                split[i] = int(Split.TEST)
        elif any(records[i].account_id in strat for i in idxs):
            positive_groups.append(key)
        else:
            plain_groups.append(key)

    pos_val, pos_test = _stratify_positive_groups(
        positive_groups, config.val_fraction, config.test_fraction
    )
    plain_val, plain_test = _assign_by_fraction(
        plain_groups, config.val_fraction, config.test_fraction
    )
    val_groups = pos_val | plain_val
    test_groups = pos_test | plain_test

    for key in positive_groups + plain_groups:
        if key in val_groups:
            s = int(Split.VAL)
        elif key in test_groups:
            s = int(Split.TEST)
        else:
            s = int(Split.TRAIN)
        for i in groups[key]:
            split[i] = s

    for i in range(n):
        if split[i] == -1:
            split[i] = int(Split.TRAIN)

    return SplitResult(
        account_ids=tuple(r.account_id for r in records),
        split=tuple(split),
    )
