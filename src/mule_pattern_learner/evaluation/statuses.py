"""What a row of a test ranking is on a Monday, by name (docs/design.md, "Rankings"): a non-mule,
or a true mule active, on its early-warning Monday, before its activity or after it.

This module holds the five words and nothing else. evaluation.answer_key reads the truth and
picks one word for each account and Monday; the words live here so that the benchmark, which
never reads the truth, can build a synthetic ranking from them without importing the answer
key's module.
"""

from __future__ import annotations

NON_MULE = "non_mule"
ACTIVE = "active"
EARLY_WARNING = "early_warning"
BEFORE = "before"
AFTER = "after"
