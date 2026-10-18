"""The keyed draws (docs/design.md, "Training world" and "If whole-world scoring is too slow").

Every (account, Monday) has one keyed hash: blake2b, keyed by the draw key of config and 8 bytes
long, of "<account id>|<Monday in epoch milliseconds>", read as a big-endian integer below
2^64. tests/data/test_draw.py pins its outputs.

- The training draw of a Monday takes the n accounts of its pool with the lowest hashes (ties,
  which a 64-bit hash practically never has, by account id). So it is uniform, and it nests:
  when a few accounts join or leave the pool, the rest of the draw stays, and with it their
  cached rows.
- The fallback sample of a Monday takes each account whose hash is below 2^64 divided by the
  sample's weight, so each is included with chance 1 over the weight, and stands for that many.

The training draw and the fallback sample read the same hash, so in a world that trains in
one rotation and selects in another, the drawn accounts are cached for both.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import heapq

# The bytes of each hash, and the bound every hash lies below.
DIGEST_BYTES = 8
SCALE = 2 ** (8 * DIGEST_BYTES)


def draw_hash(key: bytes, account_id: str, t_ms: int) -> int:
    """The keyed hash of an account on the Monday t."""
    message = f"{account_id}|{t_ms}".encode()
    digest = hashlib.blake2b(message, key=key, digest_size=DIGEST_BYTES).digest()
    return int.from_bytes(digest, "big")


def lowest(key: bytes, accounts: Iterable[str], t_ms: int, n: int) -> list[str]:
    """The n accounts with the lowest keyed hashes on the Monday t, ties by id, in that order;
    every account when there are at most n."""
    if n < 0:
        raise ValueError(f"A draw takes a number of accounts that is not negative, got {n}")
    ranked = ((draw_hash(key, account, t_ms), account) for account in accounts)
    return [account for _, account in heapq.nsmallest(n, ranked)]


def sampled(key: bytes, accounts: Iterable[str], t_ms: int, weight: int) -> list[str]:
    """The accounts the fallback sample of the Monday t includes, each with chance 1 / weight,
    in the order given."""
    if weight < 1:
        raise ValueError(f"A sample's weight is at least 1, got {weight}")
    return [account for account in accounts if draw_hash(key, account, t_ms) * weight < SCALE]
