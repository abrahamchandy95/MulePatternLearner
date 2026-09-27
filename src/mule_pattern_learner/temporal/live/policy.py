"""The frozen scope every run samples in, and the rejected-root limit."""

from __future__ import annotations

from typing import Any

# Every run keeps its splits in disjoint scope partitions (strict inductive); outputs
# record it so their performance claims say what they cover.
EVALUATION_PROTOCOL = "strict_inductive"


def context_scope(config: dict[str, Any]) -> str:
    """The scope_id of every context of a run; strict inductive splits need one."""
    scope = config.get("scope_id")
    if not scope:
        raise ValueError("Strict inductive sampling requires a frozen TigerGraph scope_id")
    return str(scope)


def exceeds_rejection_limit(rejected: int, positives: int, total: int, limit: float) -> bool:
    """Whether rejected roots censor a set: any observed positive, or over limit * total."""
    return bool(positives) or rejected > limit * total
