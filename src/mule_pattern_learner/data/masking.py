"""
Positive-Unlabeled (PU) label masking and leakage-safe splitting.

THE PROBLEM THIS SOLVES
-----------------------
PhantomLedger gives perfectly-labeled data: we know every mule. Real-world
fraud data does not — only a small fraction of mules are ever discovered, and
whole rings can go completely unnoticed. To make experiments meaningful we
*simulate* the real-world condition: hide most of the true mule labels, train on
the crippled labels, and evaluate against the full truth (how many hidden mules
did the model surface?).

Two distinct label views, which must never contaminate each other:
  * pu_label   — the masked training signal. 1 = revealed mule (a known
                 positive); 0 = unlabeled (could be an undiscovered mule,
                 including a whole dark ring). This is what the model trains on.
  * true_label — the full ground truth. Used ONLY for evaluation, never as a
                 training input.

WHY RING-AWARE MASKING (not uniform random)
-------------------------------------------
Real fraud-label discovery is clustered, not uniform:
  * Whole rings go undetected. If we masked uniformly, every ring would keep
    ~p% of its members labeled, so NO ring is ever fully dark — defeating the
    disconnected-ring generalization test that is the whole point of inductive
    PU detection. So we make a fraction of rings FULLY DARK (zero revealed
    labels): the model must surface them with no positive examples nearby.
  * The rest are PARTIALLY REVEALED: a few members known, the rest hidden —
    "we caught a couple people in this ring."
Globally the revealed positives total ~`reveal_prevalence` of all mules.

This produces three account buckets (recorded for stratified evaluation):
  * REVEALED_POS  — mule, label kept (pu_label = 1)
  * HIDDEN_POS    — mule, label masked (pu_label = 0); the model should rank
                    these high. HIDDEN_POS in dark rings are the hardest cases.
  * UNLABELED_NEG — non-mule (pu_label = 0). Mostly-but-not-certainly clean.

LEAKAGE-SAFE SPLITTING
----------------------
Two interacting controls, both enforced here:
  * Party-grouped split: all accounts owned by one Party go to the SAME split.
    Otherwise an owner's accounts straddle train/test and leak through
    Party_Has_Account.
  * Dark rings concentrated in TEST: a dark ring is only a fair test of
    "find an unseen ring" if the model never trained on any of its members.
    We therefore assign whole dark rings to the test split.

This module is PURE (no TigerGraph / torch imports) so the masking algorithm is
unit-testable in isolation. The graph I/O wrapper lives separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
import random


class Bucket(IntEnum):
    """Per-account masking bucket, recorded for stratified evaluation."""

    UNLABELED_NEG = 0  # non-mule, pu_label 0
    REVEALED_POS = 1  # mule, label kept (pu_label 1)
    HIDDEN_POS = 2  # mule, label masked (pu_label 0) — target of detection


class Split(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2


@dataclass(frozen=True, slots=True)
class MaskConfig:
    """Parameters for ring-aware PU masking + split. Seeded for reproducibility.

    reveal_prevalence:
        Target fraction of ALL mules that remain labeled (revealed positives).
        ~0.04 models "under 4% of fraudsters are known".
    dark_ring_fraction:
        Fraction of mule-containing rings that are made FULLY DARK (zero
        revealed labels). These go to the test split.
    val_fraction / test_fraction:
        Party-grouped split proportions (by Party, not by account). The
        remainder is train. Dark-ring accounts are forced into test on top of
        this (so effective test size may exceed test_fraction slightly).
    seed:
        RNG seed; the whole assignment is deterministic given the inputs.
    """

    reveal_prevalence: float = 0.04
    dark_ring_fraction: float = 0.30
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 1337

    def validate(self) -> None:
        if not (0.0 < self.reveal_prevalence <= 1.0):
            raise ValueError("reveal_prevalence must be in (0, 1].")
        if not (0.0 <= self.dark_ring_fraction <= 1.0):
            raise ValueError("dark_ring_fraction must be in [0, 1].")
        if self.val_fraction < 0.0 or self.test_fraction < 0.0:
            raise ValueError("split fractions must be non-negative.")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1.")


@dataclass(frozen=True, slots=True)
class AccountRecord:
    """One account's ground-truth inputs to masking.

    account_id:  the Account primary id.
    is_mule:     true label (is_fraud == 1).
    party_id:    owning Party id, or None if unowned (e.g. external/merchant).
    ring_id:     true fraud-ring id (>0) from the raw ledger, or 0 if none.
    """

    account_id: str
    is_mule: bool
    party_id: str | None
    ring_id: int


@dataclass(frozen=True, slots=True)
class MaskResult:
    """Per-account masking + split outputs, aligned to the input order."""

    account_ids: tuple[str, ...]
    pu_label: tuple[int, ...]  # 1 = revealed positive, else 0
    true_label: tuple[int, ...]  # 1 = mule (ground truth, eval only)
    bucket: tuple[int, ...]  # Bucket value
    split: tuple[int, ...]  # Split value

    def summary(self) -> dict[str, int]:
        n = len(self.account_ids)
        revealed = sum(1 for b in self.bucket if b == Bucket.REVEALED_POS)
        hidden = sum(1 for b in self.bucket if b == Bucket.HIDDEN_POS)
        neg = sum(1 for b in self.bucket if b == Bucket.UNLABELED_NEG)
        return {
            "accounts": n,
            "true_mules": revealed + hidden,
            "revealed_positives": revealed,
            "hidden_positives": hidden,
            "unlabeled_negatives": neg,
            "train": sum(1 for s in self.split if s == Split.TRAIN),
            "val": sum(1 for s in self.split if s == Split.VAL),
            "test": sum(1 for s in self.split if s == Split.TEST),
        }


def _group_accounts_by_party(
    records: Sequence[AccountRecord],
) -> dict[str, list[int]]:
    """Map party_id -> indices. Unowned accounts get a synthetic singleton
    group keyed by their own id so they split independently."""
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        key = r.party_id if r.party_id is not None else f"__solo__{r.account_id}"
        groups.setdefault(key, []).append(i)
    return groups


def _ring_to_mule_indices(
    records: Sequence[AccountRecord],
) -> dict[int, list[int]]:
    """Map ring_id (>0) -> indices of its MULE accounts."""
    rings: dict[int, list[int]] = {}
    for i, r in enumerate(records):
        if r.ring_id > 0 and r.is_mule:
            rings.setdefault(r.ring_id, []).append(i)
    return rings


def resolve_account_rings(
    txn_endpoints: Sequence[tuple[str, int]],
) -> dict[str, int]:
    """Resolve each account's true ring_id from raw ledger transactions.

    Input: a sequence of (account_id, ring_id) pairs — one per (transaction,
    endpoint) appearance, i.e. each transaction contributes its src and its dst.
    ring_id == 0 means the transaction is not ring activity (legitimate / cover
    traffic).

    Rule (researched + reasoned for PhantomLedger semantics):
      An account's ring_id is the FIRST NONZERO ring_id among its transactions.
      Rationale:
        * A mule launders for a ring, but also makes legitimate "cover"
          transactions stamped ring_id=0. The zeros are camouflage, not evidence
          of "no ring", so they must be ignored — NOT treated as a majority
          vote (which would wrongly call a mule ringless).
        * Accounts almost always belong to a single ring in the generator;
          the rare multi-ring account is assigned deterministically to its
          first-seen nonzero ring (stable given input order) rather than
          dropped, so we never discard a true ring mule.
      An account with only ring_id=0 transactions resolves to 0 (no ring).

    Note: this returns ring_id for ALL accounts seen; the masking core uses it
    only for MULE accounts (a victim with an incidental ring-stamped inbound
    transfer is not a ring member for dark-ring purposes — see
    _ring_to_mule_indices).
    """
    out: dict[str, int] = {}
    for account_id, ring_id in txn_endpoints:
        if account_id not in out:
            out[account_id] = 0
        if ring_id > 0 and out[account_id] == 0:
            out[account_id] = ring_id
    return out


def apply_pu_mask(records: Sequence[AccountRecord], config: MaskConfig) -> MaskResult:
    """Produce PU labels, masking buckets, and a leakage-safe split.

    Deterministic given (records, config). Algorithm:

      1. Identify mule-containing rings. Choose `dark_ring_fraction` of them to
         be fully dark (no revealed labels); their accounts are forced to TEST.
      2. From the NON-dark mules, reveal a random subset sized so the global
         revealed count ≈ reveal_prevalence * total_mules. (Dark-ring mules are
         never revealed, so they count against the hidden pool.)
      3. Party-grouped split for all non-forced accounts (val/test/train by
         party). Dark-ring accounts already pinned to TEST.
    """
    config.validate()
    rng = random.Random(config.seed)

    n = len(records)
    total_mules = sum(1 for r in records if r.is_mule)

    pu = [0] * n
    true = [1 if r.is_mule else 0 for r in records]
    bucket = [int(Bucket.UNLABELED_NEG)] * n
    split = [-1] * n  # -1 = unassigned

    # ── Step 1: choose dark rings, pin their accounts to TEST ──
    ring_mules = _ring_to_mule_indices(records)
    ring_ids = sorted(ring_mules.keys())
    rng.shuffle(ring_ids)
    n_dark = int(round(config.dark_ring_fraction * len(ring_ids)))
    dark_rings: set[int] = set(ring_ids[:n_dark])

    dark_mule_indices: set[int] = set()
    for rid in dark_rings:
        for idx in ring_mules[rid]:
            dark_mule_indices.add(idx)

    # All accounts belonging to a dark ring (mules AND any non-mule members)
    # are pinned to TEST so the model never trains on a dark ring.
    for i, r in enumerate(records):
        if r.ring_id in dark_rings:
            split[i] = int(Split.TEST)

    # ── Step 2: reveal a subset of NON-dark mules to hit global prevalence ──
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

    # ── Step 3: Party-grouped split for the not-yet-assigned accounts ──
    groups = _group_accounts_by_party(records)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    # A party is "free" only if NONE of its accounts were pinned to TEST in
    # step 1; a party touching a dark ring goes entirely to TEST (keeps owners
    # from straddling).
    free_groups: list[str] = []
    for key in group_keys:
        idxs = groups[key]
        if any(split[i] == int(Split.TEST) for i in idxs):
            for i in idxs:
                split[i] = int(Split.TEST)
        else:
            free_groups.append(key)

    n_free = len(free_groups)
    n_val = int(round(config.val_fraction * n_free))
    n_test = int(round(config.test_fraction * n_free))
    val_groups = set(free_groups[:n_val])
    test_groups = set(free_groups[n_val : n_val + n_test])

    for key in free_groups:
        if key in val_groups:
            s = int(Split.VAL)
        elif key in test_groups:
            s = int(Split.TEST)
        else:
            s = int(Split.TRAIN)
        for i in groups[key]:
            split[i] = s

    # safety: no account left unassigned
    for i in range(n):
        if split[i] == -1:
            split[i] = int(Split.TRAIN)

    return MaskResult(
        account_ids=tuple(r.account_id for r in records),
        pu_label=tuple(pu),
        true_label=tuple(true),
        bucket=tuple(bucket),
        split=tuple(split),
    )
