"""Configurations saved before the typed configuration, converted to RunConfig.

A model saved before RunConfig holds a flat table of the old setting names instead of
RunConfig.to_dict(). converted_run_config converts it through SAVED_SETTINGS, the one
table from old names to RunConfig fields (and SAVED_VALUES, the values that name
something else now), so such models load and score as they did.

The conversion stays until the new baseline run has a ground-truth audit, so that the
models trained before the restructure can still be compared with it, and goes when main
is replaced.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..config import DEFAULT_CONFIG, RunConfig
from ..contract.feature_groups import BUILT_IN_GROUPS
from ..contract.sampler_plan import PoolPlan

# Where each setting of a configuration saved before the typed configuration lives in
# RunConfig: the dotted name of its field, or None for a setting that names nothing
# now. Its [sampler] table held the roots pool beside the sampler's own settings, and
# a [sampler.children] table of changes to the roots pool without associations.
SAVED_SETTINGS: dict[str, str | None] = {
    # The graph identity is the source id the prepared dataset records, and a dataset
    # is found by its run, so neither is a setting.
    "dataset_id": None,
    "prepared_id": None,
    "scope_id": "scope.id",
    "create_scope": "scope.create",
    "scope_unowned": "scope.unowned",
    "reveal_per_split": "scope.reveal_per_split",
    "reveal_salt": "scope.reveal_salt",
    "dates": "dataset.dates",
    "seed_limits": "dataset.seed_limits",
    "cohort_seed": "dataset.seed",
    "split_seed": "dataset.split_seed",
    "fanouts": "sampler.fanouts",
    "sampler.recent": "sampler.roots.recent",
    "sampler.older": "sampler.roots.older",
    "sampler.distinct": "sampler.roots.distinct",
    "sampler.associations": "sampler.roots.associations",
    "sampler.max_history": "sampler.roots.max_history",
    "sampler.children.recent": "sampler.children.recent",
    "sampler.children.older": "sampler.children.older",
    "sampler.children.distinct": "sampler.children.distinct",
    "sampler.children.associations": "sampler.children.associations",
    "sampler.children.max_history": "sampler.children.max_history",
    "sampler.relation_fanouts": "sampler.relation_fanouts",
    "sampler.association_fanout": "sampler.association_fanout",
    "sampler.association_slots": "sampler.association_slots",
    "sampler.backend": "sampler.backend",
    "sampler.evaluation_seed": "sampler.evaluation_seed",
    "feature_groups": "features",
    "architecture": "model.architecture",
    "hidden": "model.hidden",
    "heads": "model.heads",
    "dropout": "model.dropout",
    "slot_sum": "model.slot_sum",
    "class_prior": "loss.class_prior",
    "positive_weight": "loss.positive_weight",
    "seed": "training.seed",
    "epochs": "training.epochs",
    "steps_per_epoch": "training.steps_per_epoch",
    "batch_size": "training.batch_size",
    "patience": "training.patience",
    "learning_rate": "training.learning_rate",
    "weight_decay": "training.weight_decay",
    "weight_average_decay": "training.weight_average_decay",
    "evaluation_unlabeled_limit": "training.proxy_unlabeled_limit",
    "request_batch_size": "transport.request_batch_size",
    "query_concurrency": "transport.query_concurrency",
    "context_lru_capacity": "transport.context_lru_capacity",
    "encoding_check_every": "transport.encoding_check_every",
    "max_query_attempts": "transport.max_query_attempts",
    "max_outage_s": "transport.max_outage_s",
    # The batch size of the removed SQLite context storage.
    "prepare_batch_size": None,
    "device": "runtime.device",
    "threads": "runtime.threads",
    "deterministic": "runtime.deterministic",
    "prefetch_batches": "runtime.prefetch_batches",
    "checkpoint_every_steps": "runtime.checkpoint_every_steps",
    "log_every_steps": "runtime.log_every_steps",
    "max_rejected_root_fraction": "runtime.max_rejected_root_fraction",
    # Settings of removed paths, saved with the one value this code implements
    # (RETIRED_VALUES), and the extraction groups, which only widened the request.
    "context_storage": None,
    "evaluation_protocol": None,
    "label_policy": None,
    "observed_labels": None,
    "sampler.policy": None,
    "extraction_groups": None,
    # The model variants, which became settings (converted_run_config).
    "variant": None,
}
# Saved values that name something else now, by the old setting name: the graph model's
# architecture was "split" before it became TGAT.
SAVED_VALUES: dict[str, dict[object, object]] = {"architecture": {"split": "tgat"}}
# Saved settings whose null value took a default: the reveal's built-in budget, and the
# training seed for the reservoir seed and the reveal salt.
SEEDED_DEFAULTS = frozenset({"reveal_per_split", "reveal_salt", "cohort_seed"})
# Saved settings that the old code filled in when a configuration left them absent or
# null, with values the built-in run does not share, and that nothing else guards: its
# fan-outs were (8, 4) and its sampler pools its own. A configuration without either is
# refused. Other settings the old code filled in differently are guarded (strict
# weight loading refuses another slot sum or architecture, the input fingerprint other
# feature groups) or act only in training (the weight average and the positive weight).
REQUIRED_SETTINGS = ("fanouts", "sampler")
RETIRED_VALUES: dict[str, object] = {
    "context_storage": "stream",
    "evaluation_protocol": "strict_inductive",
    "label_policy": "graph_observed",
    "observed_labels": None,
    "sampler.policy": "resample",
}


def _flat(table: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """A saved table's settings by dotted name; only the sampler tables nest."""
    flat: dict[str, Any] = {}
    for key, value in table.items():
        name = prefix + key
        if name in ("sampler", "sampler.children") and isinstance(value, dict):
            flat.update(_flat(value, name + "."))
        else:
            flat[name] = value
    return flat


def converted_run_config(saved: dict[str, Any]) -> RunConfig:
    """The RunConfig of a flat table of old setting names.

    The table goes through SAVED_SETTINGS, with the values SAVED_VALUES renames. It
    must hold the REQUIRED_SETTINGS, which every model the old code saved does. A
    setting it leaves absent takes the built-in run's value, except where the old code
    gave it another rule, which this follows: a [sampler] table's absent pool keys are
    those of PoolPlan() and its children pool is the roots pool without associations,
    changed by [sampler.children]; an absent or null reservoir seed and reveal salt are
    the training seed, and a null reveal budget is the built-in one. The variant
    "tabular" is the summary architecture and "no_fourier" the feature groups without
    time_encoding.
    """
    flat = _flat(saved)
    unknown = sorted(set(flat) - SAVED_SETTINGS.keys())
    if unknown:
        raise ValueError(f"Unknown saved configuration key(s): {', '.join(unknown)}")
    for name, value in RETIRED_VALUES.items():
        if name in flat and flat[name] != value:
            remains = "" if value is None else f": only {value!r} remains"
            raise ValueError(f"{name} = {flat[name]!r} is no longer supported{remains}")
    missing = [name for name in REQUIRED_SETTINGS if saved.get(name) is None]
    if missing:
        raise ValueError(
            f"The saved configuration names no {' or '.join(missing)}; the code that saved "
            "it assumed values other than the built-in run's, so it cannot be scored as trained"
        )
    table: dict[str, Any] = {}
    for name, value in flat.items():
        target = SAVED_SETTINGS[name]
        if target is None or (value is None and name in SEEDED_DEFAULTS):
            continue
        if name in SAVED_VALUES:
            value = SAVED_VALUES[name].get(value, value)
        *sections, key = target.split(".")
        node = table
        for section in sections:
            node = node.setdefault(section, {})
        node[key] = value
    if isinstance(saved.get("sampler"), dict):
        sampler = table.setdefault("sampler", {})
        roots = sampler["roots"] = {**asdict(PoolPlan()), **sampler.get("roots", {})}
        sampler["children"] = {**roots, "associations": 0, **sampler.get("children", {})}
    seed = flat.get("seed", DEFAULT_CONFIG.training.seed)
    table.setdefault("dataset", {}).setdefault("seed", seed)
    table.setdefault("scope", {}).setdefault("reveal_salt", seed)
    variant = flat.get("variant", "temporal")
    if variant == "tabular":
        table.setdefault("model", {})["architecture"] = "summary"
    elif variant == "no_fourier":
        groups = table.get("features") or BUILT_IN_GROUPS
        table["features"] = [group for group in groups if group != "time_encoding"]
    elif variant != "temporal":
        raise ValueError(f"variant = {variant!r} is no longer supported")
    return RunConfig.from_dict(table)
