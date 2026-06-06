from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
import random


class Bucket(IntEnum):
    """Per-account masking bucket."""

    UNLABELED_NEG = 0
    REVEALED_POS = 1
    HIDDEN_POS = 2


@dataclass(frozen=True, slots=True)
class MaskConfig:
    """Parameters for ring-aware PU masking.

    reveal_prevalence:
        Target fraction of ALL mules that remain labeled (pu_label = 1).
        ~0.04 models "under 4% of fraudsters are known".
    dark_ring_fraction:
        Fraction of mule-containing rings made FULLY DARK (no revealed labels).
        Their accounts are emitted as forced-test (see MaskResult.forced_test).
    seed:
        RNG seed; the assignment is deterministic given inputs and seed.
    """

    reveal_prevalence: float = 0.04
    dark_ring_fraction: float = 0.30
    seed: int = 1337

    def validate(self) -> None:
        if not (0.0 < self.reveal_prevalence <= 1.0):
            raise ValueError("reveal_prevalence must be in (0, 1].")
        if not (0.0 <= self.dark_ring_fraction <= 1.0):
            raise ValueError("dark_ring_fraction must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class MaskRecord:
    """Per-account input for masking: identity, truth, and ring membership."""

    account_id: str
    is_mule: bool
    ring_id: int


@dataclass(frozen=True, slots=True)
class MaskResult:
    """Per-account masking outputs, aligned to the input order.

    forced_test is the set of dark-ring account ids that must be held out of
    training. It is the explicit handoff to the splitter: masking decides which
    rings are fully dark, and the caller passes forced_test to split_accounts.
    Masking itself performs no train/val/test assignment.
    """

    account_ids: tuple[str, ...]
    pu_label: tuple[int, ...]
    true_label: tuple[int, ...]
    bucket: tuple[int, ...]
    forced_test: frozenset[str]

    def summary(self) -> dict[str, int]:
        revealed = sum(1 for b in self.bucket if b == Bucket.REVEALED_POS)
        hidden = sum(1 for b in self.bucket if b == Bucket.HIDDEN_POS)
        neg = sum(1 for b in self.bucket if b == Bucket.UNLABELED_NEG)
        return {
            "accounts": len(self.account_ids),
            "true_mules": revealed + hidden,
            "revealed_positives": revealed,
            "hidden_positives": hidden,
            "unlabeled_negatives": neg,
            "forced_test_accounts": len(self.forced_test),
        }


def _ring_to_mule_indices(
    records: Sequence[MaskRecord],
) -> dict[int, list[int]]:
    rings: dict[int, list[int]] = {}
    for i, r in enumerate(records):
        if r.ring_id > 0 and r.is_mule:
            rings.setdefault(r.ring_id, []).append(i)
    return rings


def resolve_account_rings(
    txn_endpoints: Sequence[tuple[str, int]],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for account_id, ring_id in txn_endpoints:
        if account_id not in out:
            out[account_id] = 0
        if ring_id > 0 and out[account_id] == 0:
            out[account_id] = ring_id
    return out


def apply_pu_mask(records: Sequence[MaskRecord], config: MaskConfig) -> MaskResult:
    """
    Simulate realistic, clustered fraud-label discovery on synthetic data.

    Algorithm:
      1. Identify mule-containing rings. Choose dark_ring_fraction of them to be
         fully dark (no revealed labels). Their accounts are emitted in
         forced_test (to be pinned to TEST by the splitter), so the model never
         trains on a fully-undiscovered ring.
      2. From the NON-dark mules, reveal a random subset sized so the global
         revealed count is about reveal_prevalence * total_mules.
    """
    config.validate()
    rng = random.Random(config.seed)

    n = len(records)
    total_mules = sum(1 for r in records if r.is_mule)

    pu: list[int] = [0] * n
    true: list[int] = [1 if r.is_mule else 0 for r in records]
    bucket: list[int] = [int(Bucket.UNLABELED_NEG)] * n

    ring_mules = _ring_to_mule_indices(records)
    ring_ids = sorted(ring_mules.keys())
    rng.shuffle(ring_ids)
    n_dark = int(round(config.dark_ring_fraction * len(ring_ids)))
    dark_rings: set[int] = set(ring_ids[:n_dark])

    dark_mule_indices: set[int] = set()
    for rid in dark_rings:
        for idx in ring_mules[rid]:
            dark_mule_indices.add(idx)

    forced_test: set[str] = {
        records[i].account_id for i, r in enumerate(records) if r.ring_id in dark_rings
    }

    target_revealed = int(round(config.reveal_prevalence * total_mules))
    revealable: list[int] = [
        i for i, r in enumerate(records) if r.is_mule and i not in dark_mule_indices
    ]
    rng.shuffle(revealable)
    revealed_set: set[int] = set(revealable[:target_revealed])

    for i, r in enumerate(records):
        if r.is_mule:
            if i in revealed_set:
                pu[i] = 1
                bucket[i] = int(Bucket.REVEALED_POS)
            else:
                pu[i] = 0
                bucket[i] = int(Bucket.HIDDEN_POS)
        else:
            pu[i] = 0
            bucket[i] = int(Bucket.UNLABELED_NEG)

    return MaskResult(
        account_ids=tuple(r.account_id for r in records),
        pu_label=tuple(pu),
        true_label=tuple(true),
        bucket=tuple(bucket),
        forced_test=frozenset(forced_test),
    )
