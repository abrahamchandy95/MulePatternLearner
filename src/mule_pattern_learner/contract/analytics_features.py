"""The feature groups the analytics context query computes and training never reads.

Training keeps only the groups of the built-in run (the owner's decision in
docs/architecture.md). These are the others: account aggregates over the whole visible history
(age, windows, ratios, recency, association counts, decayed sums, identity order) and
the pair window counts and device and IP ages of sampled messages. They are not in
contract.feature_groups.FEATURE_GROUPS, so no FeaturePlan can name them, and no
training module imports this one (an import contract). tigergraph.render renders
gsql/analytics/analytics_context.gsql from them, for analyses such as `mule diagnose`.
"""

from __future__ import annotations

from .feature_groups import FeatureGroup
from .graph_schema import ASSOCIATIONS

WINDOWS = {"1h": 3_600_000, "1d": 86_400_000, "7d": 604_800_000, "30d": 2_592_000_000}
AMOUNT_RATIO_WINDOWS = ("1d", "7d")
AMOUNT_RATIO_FLOOR = 1.0
AMOUNT_RATIO_CAP = 100.0
AMOUNT_RATIO_FEATURES = tuple(f"{window}_out_in_amount_ratio" for window in AMOUNT_RATIO_WINDOWS)
ROLLING_FIELDS = (
    "out_count",
    "in_count",
    "out_amount",
    "in_amount",
    "out_missing",
    "in_missing",
    "out_zelle",
    "in_zelle",
    "out_unique",
    "in_unique",
)
HALF_LIVES = {"1d": 86_400_000, "7d": 604_800_000, "30d": 2_592_000_000, "90d": 7_776_000_000}
# The associations whose starts and ends identity_order counts over the last ten events.
IDENTITY_ORDER_RELATIONS = (
    "Account_Owned_By_Party",
    "Account_Bound_From_Token",
    "Account_Uses_Device",
)
# The windows of the pair counts before each sampled event.
PAIR_WINDOWS = {"1h": 3_600_000, "1d": 86_400_000, "7d": 604_800_000}

ANALYTICS_GROUPS = {
    "entity_age": FeatureGroup("node", ("age_days",)),
    "history_support": FeatureGroup(
        "summary", ("visible_event_count", "history_lt_5_events"), ("history_lt_5_events",)
    ),
    "rolling_windows": FeatureGroup(
        "summary", tuple(f"{w}_{f}" for w in WINDOWS for f in ROLLING_FIELDS)
    ),
    "amount_ratios": FeatureGroup("summary", AMOUNT_RATIO_FEATURES, requires=("rolling_windows",)),
    "recency": FeatureGroup(
        "summary",
        ("out_recency_days", "in_recency_days", "out_recency_present", "in_recency_present"),
    ),
    "association_counts": FeatureGroup(
        "summary",
        tuple(
            f"{r}_{state}" for pair in ASSOCIATIONS for r in pair for state in ("active", "ended")
        ),
    ),
    "decayed_activity": FeatureGroup(
        "summary",
        tuple(
            f"decay_{h}_{d}_{v}"
            for h in HALF_LIVES
            for d in ("out", "in")
            for v in ("count", "amount")
        ),
    ),
    "identity_order": FeatureGroup(
        "summary",
        tuple(
            f"{r}_{state}_last10" for r in IDENTITY_ORDER_RELATIONS for state in ("starts", "ends")
        ),
    ),
    "pair_window_counts": FeatureGroup(
        "message", tuple(f"pair_count_{window}" for window in PAIR_WINDOWS)
    ),
    "device_ip_context": FeatureGroup(
        "message",
        ("device_age_seconds", "device_present", "ip_age_seconds", "ip_present"),
        ("device_present", "ip_present"),
    ),
}
