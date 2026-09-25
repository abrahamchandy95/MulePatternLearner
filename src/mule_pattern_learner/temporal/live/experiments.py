"""A label-blind ablation matrix. It describes runs; it does not start training."""

from copy import deepcopy
from typing import Any

from .contract import DEFAULT_GROUPS, LEGACY_GROUPS, FeaturePlan, extraction_plan


def feature_experiments(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Keep dates, scope, revealed labels, prior and extraction fixed across feature arms.

    Every arm keeps the base ``extraction_groups`` (training compares it with the
    preparation), so all arms can train on one streamed dataset prepared with that
    superset, whatever their architecture: the hop-2 flags follow each arm's model.
    A SQLite cache also records the extraction architecture, so there only arms of
    the prepared architecture fit. An arm needing a group outside the superset
    raises here. train() additionally checks that the source covers each arm's
    inputs at both hops.
    """
    cases = {
        "legacy_single": (LEGACY_GROUPS, "single"),
        "legacy_split": (LEGACY_GROUPS, "split"),
        "windows_only": (
            ("entity_meta", "rolling_windows", "amount_ratios", "recency", "association_counts"),
            "summary",
        ),
        "decay_only": (("entity_meta", "decayed_activity", "history_support"), "summary"),
        "event_core": (("entity_meta", "message_core", "time_encoding"), "split"),
        "zero_node": (("message_core", "time_encoding"), "split"),
        "pair_history": (("entity_meta", "message_core", "time_encoding", "pair_history"), "split"),
        "flow_timing": (("entity_meta", "message_core", "time_encoding", "flow_timing"), "split"),
        "channel": (("entity_meta", "message_core", "time_encoding", "event_channel"), "split"),
        "window_free_combined": (DEFAULT_GROUPS, "split"),
        "window_free_decay": ((*DEFAULT_GROUPS, "decayed_activity", "history_support"), "split"),
        "window_free_identity_order": ((*DEFAULT_GROUPS, "identity_order"), "split"),
        "window_free_device_ip": ((*DEFAULT_GROUPS, "device_ip_context"), "split"),
    }
    result = {}
    for name, (groups, architecture) in cases.items():
        config = deepcopy(base)
        config.update(feature_groups=list(groups), architecture=architecture, variant="temporal")
        FeaturePlan.from_config(config)
        try:
            extraction_plan(config)
        except ValueError as error:
            raise ValueError(f"Arm {name} does not fit extraction_groups: {error}") from None
        result[name] = config
    return result
