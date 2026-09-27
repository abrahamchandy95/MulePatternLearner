"""A label-blind ablation matrix. It describes runs; it does not start training."""

from copy import deepcopy
from typing import Any

from ...contract.feature_groups import BUILT_IN_GROUPS, DEFAULT_GROUPS, POOL_GROUPS, FeaturePlan


def feature_experiments(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Keep dates, scope, revealed labels and prior fixed across feature arms.

    Feature groups are not a preparation setting, so all arms train on one prepared
    cohort whatever their groups and architecture: the source requests each arm's
    groups and hop-2 flags. train() checks that the source covers each arm's inputs at
    both hops.

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
        result[name] = config
    return result
