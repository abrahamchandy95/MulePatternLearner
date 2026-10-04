"""The label reveal's draws and parameter defaults, shared with its GSQL.

The reveal (gsql/queries/label_reveal.gsql) simulates when a bank would have discovered
each mule from uniforms its draw query returns. reveal_uniforms is that query's modular
mixer, exactly, and REVEAL_DEFAULTS are the defaults the reveal query declares (a test
compares them with the GSQL). tigergraph.reveal runs the job with a run's budget and
salt; the job's Python mirror (reference.label_reveal) reads both from here, so the
mirror needs no adapter.
"""

from __future__ import annotations

# The defaults the reveal query declares. A run sends the budget and salt
# (tigergraph.reveal.reveal_parameters); the model parameters stay the query's own. The
# budget's default, the most the reveal takes per split, is what a run sends to reveal
# every discovered mule.
REVEAL_DEFAULTS: dict[str, float] = {
    "budget": 1000,
    "salt": 42,
    "p_report": 0.65,
    "p_action_first": 0.5,
    "p_action_later": 0.7,
    "proactive_per_day": 0.00045,
    "trace_probability": 0.25,
    "propensity_slope": 1.0,
    "propensity_floor": 0.05,
}
_PRIME = 2147483647


def reveal_uniforms(key: int, salt: int, n: int) -> list[float]:
    """The uniforms the reveal's draw query returns (same modular mixer, exactly)."""
    values = []
    for i in range(1, n + 1):
        x = ((key % _PRIME) + i * 1000003 + (salt % _PRIME) * 7919) % _PRIME
        x = (x * 48271 + 11) % _PRIME
        y = (x * x + 1013904223) % _PRIME
        x = (y * 69621 + x) % _PRIME
        y = (x * x + 12345) % _PRIME
        x = (y * 48271 + x) % _PRIME
        values.append((x + 0.5) / _PRIME)
    return values
