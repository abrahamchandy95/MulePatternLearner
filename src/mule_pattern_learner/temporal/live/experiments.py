"""A label-blind ablation matrix. It describes runs; it does not start training."""

from copy import deepcopy
from typing import Any

from .contract import BUILT_IN_GROUPS, DEFAULT_GROUPS, POOL_GROUPS, FeaturePlan, extraction_plan


def feature_experiments(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Keep dates, scope, revealed labels, prior and extraction fixed across feature arms.

    Every arm keeps the base ``extraction_groups`` (training compares it with the
    preparation), so all arms can train on one dataset prepared with that superset,
    whatever their architecture: the hop-2 flags follow each arm's model. An arm
    needing a group outside the superset raises here. train() additionally checks that the source covers each arm's
    inputs at both hops.

    Every arm sets ``slot_sum``. The feature-group arms keep it off, the model they
    were designed on. The ``built_in`` arms measure the built-in run's own additions:
    the slot sum, the pool groups, the internal-payer counts alone, and a summary
    model on the same root inputs (the tabular control, no attention or slots).
    """
    cases = {
        "windows_only": (
            ("entity_meta", "rolling_windows", "amount_ratios", "recency", "association_counts"),
            "summary",
            False,
        ),
        "decay_only": (("entity_meta", "decayed_activity", "history_support"), "summary", False),
        "event_core": (("entity_meta", "message_core", "time_encoding"), "split", False),
        "zero_node": (("message_core", "time_encoding"), "split", False),
        "pair_history": (
            ("entity_meta", "message_core", "time_encoding", "pair_history"),
            "split",
            False,
        ),
        "flow_timing": (
            ("entity_meta", "message_core", "time_encoding", "flow_timing"),
            "split",
            False,
        ),
        "channel": (
            ("entity_meta", "message_core", "time_encoding", "event_channel"),
            "split",
            False,
        ),
        "window_free_combined": (DEFAULT_GROUPS, "split", False),
        "window_free_decay": (
            (*DEFAULT_GROUPS, "decayed_activity", "history_support"),
            "split",
            False,
        ),
        "window_free_identity_order": ((*DEFAULT_GROUPS, "identity_order"), "split", False),
        "window_free_device_ip": ((*DEFAULT_GROUPS, "device_ip_context"), "split", False),
        "built_in": (BUILT_IN_GROUPS, "split", True),
        "built_in_no_slot_sum": (BUILT_IN_GROUPS, "split", False),
        "built_in_no_pool": (
            tuple(g for g in BUILT_IN_GROUPS if g not in POOL_GROUPS),
            "split",
            True,
        ),
        "built_in_no_internal": (
            tuple(g for g in BUILT_IN_GROUPS if g != "pool_internal_inflows"),
            "split",
            True,
        ),
        "built_in_tabular": (BUILT_IN_GROUPS, "summary", False),
    }
    result = {}
    for name, (groups, architecture, slot_sum) in cases.items():
        config = deepcopy(base)
        config.update(feature_groups=list(groups), architecture=architecture, slot_sum=slot_sum)
        FeaturePlan.from_config(config)
        try:
            extraction_plan(config)
        except ValueError as error:
            raise ValueError(f"Arm {name} does not fit extraction_groups: {error}") from None
        result[name] = config
    return result
