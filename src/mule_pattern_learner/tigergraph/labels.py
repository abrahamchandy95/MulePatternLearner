"""The graph's revealed labels for training.

TigerGraphObservedLabelReader is the label source of every run: population queries
report the revealed positives and their discovery times, never a hidden label.
"""

from __future__ import annotations

import pandas as pd

from ..data.observed_labels import align_observed_labels


class TigerGraphObservedLabelReader:
    """Observed labels paged from the graph, the label source of every run.

    The population query reports the revealed positive of the account labels
    (pu_label = 1, which the reveal writes only for a known mule it revealed) and its
    discovery time; every other account has observed_positive false and known_from_ms 0.
    """

    def read(self, metadata: pd.DataFrame) -> pd.DataFrame:
        labels = metadata[["account_id", "observed_positive", "known_from_ms"]].rename(
            columns={"observed_positive": "known_positive"}
        )
        return align_observed_labels(metadata, labels)
