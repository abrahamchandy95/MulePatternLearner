"""The maturity gap (docs/design.md, "Splits and hidden labels"), from the revealed mules
alone: the delay from onset to discovery, and the gap within which most mules are found.

A mule is revealed only when its delay is shorter than the time from its onset to the
cutoff, so the newest mules are revealed only when they were found fast, and the plain
delays of revealed mules are too short. The estimate corrects for that with the
product-limit estimator for delays observed only below a known bound (Lynden-Bell 1971;
Lagakos, Barraj and De Gruttola 1988): Kaplan-Meier run backwards from the longest delay.
No delay longer than every revealed one can be seen, so the shares are of the mules found
within the longest revealed delay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import DAY_MS

# The share of mules found within the gap, by the estimate.
DISCOVERED_SHARE = 0.9
# Float error allowed when the estimate is compared with a share.
_TOLERANCE = 1e-9


def delay_distribution(supervision: pd.DataFrame, cutoff_ms: int) -> pd.Series:
    """The estimated share of mules found within each delay from onset to discovery, by the
    delays of the mules revealed before cutoff_ms (supervision, as
    data.supervision.read_supervision gives it), indexed by delay_ms in increasing order."""
    if supervision.empty:
        raise ValueError("No revealed mule to estimate the delay from")
    onset, found = supervision.mule_onset_ms.to_numpy(), supervision.discovered_ms.to_numpy()
    if not (found < cutoff_ms).all():
        raise ValueError("Every revealed mule must have been discovered before the cutoff")
    delays = found - onset
    # Each mule was revealed because its delay was below this bound.
    bounds = cutoff_ms - onset
    values, counts = np.unique(delays, return_counts=True)
    # The mules whose delay is at most the value and could have been as long as it.
    at_risk = np.array([np.sum((delays <= value) & (value < bounds)) for value in values])
    # The share found within each delay is the product over every longer delay of the
    # share not found at it among those at risk.
    kept = 1.0 - counts / at_risk
    within = np.append(np.cumprod(kept[::-1])[::-1][1:], 1.0)
    return pd.Series(within, index=pd.Index(values, name="delay_ms"), name="discovered_share")


def implied_gap_days(
    supervision: pd.DataFrame, cutoff_ms: int, share: float = DISCOVERED_SHARE
) -> int:
    """The maturity gap the revealed mules imply: the fewest whole days within which the
    estimate finds share of the mules."""
    if not 0 < share <= 1:
        raise ValueError(f"The share of mules found must be in (0, 1], got {share}")
    within = delay_distribution(supervision, cutoff_ms)
    delay = int(within.index[within >= share - _TOLERANCE].min())
    return -(-delay // DAY_MS)
