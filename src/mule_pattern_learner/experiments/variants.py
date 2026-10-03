"""The control experiments: variants of the built-in run, the suites of them, and the seeds.

A Variant is a question and a change to a configuration. config(base, seed) applies the
change to the base run (config.DEFAULT_CONFIG, the built-in run, unless a test gives a
smaller one) and then sets training.seed alone: seeds are an axis of their own, so no
variant sets a seed, and none touches dataset.seed, dataset.split_seed or
scope.reveal_salt. Every variant therefore trains on the base run's dataset, and seed 42
of the baseline is the run `mule train` makes.

Training reads only the built-in run's feature groups (the owner's decision in
docs/architecture.md), so a variant drops groups or changes the model, the loss or the training, and
never adds a group: a group comes back into training only by moving it into the training
query on purpose. The account-activity table the retired no_graph control asked about is
a question for the diagnostics baselines, over the analytics features.

Nothing here loads torch, so scripts/run_experiments.py lists the suites and variants
under --help without it; experiments.runner.run_suite trains and compares them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import textwrap
from typing import Any

from ..config import DEFAULT_CONFIG, RunConfig, differing_settings
from ..contract.feature_groups import BUILT_IN_GROUPS, FEATURE_GROUPS, POOL_GROUPS
from ..paths import BASELINE_VARIANT

# The seeds every variant trains with; 42 is the built-in run's.
SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class Variant:
    """A named change to the base run and the question it answers."""

    name: str
    question: str
    change: Callable[[RunConfig], RunConfig]

    def config(self, base: RunConfig, seed: int) -> RunConfig:
        """The variant's configuration of base with this seed (training.seed only)."""
        return with_seed(self.change(base), seed)

    def changes(self, base: RunConfig = DEFAULT_CONFIG) -> dict[str, Any]:
        """The settings the variant changes in base, by dotted name, with their new values."""
        changed, before = self.change(base).to_dict(), base.to_dict()
        found: dict[str, Any] = {}
        for name in differing_settings(changed, before):
            value: Any = changed
            for part in name.split("."):
                value = value[part]
            found[name] = value
        return found

    def change_text(self, base: RunConfig = DEFAULT_CONFIG) -> str:
        """The changes as one line: "model.slot_sum = False; features without flow_timing"."""
        parts = []
        for name, value in self.changes(base).items():
            if name == "features":
                removed = [group for group in base.features if group not in value]
                parts.append("features without " + ", ".join(removed))
            else:
                parts.append(f"{name} = {value}")
        return "; ".join(parts) or "no change"


def with_training(config: RunConfig, **changes: Any) -> RunConfig:
    return replace(config, training=replace(config.training, **changes))


def with_seed(config: RunConfig, seed: int) -> RunConfig:
    return with_training(config, seed=seed)


def with_model(config: RunConfig, **changes: Any) -> RunConfig:
    return replace(config, model=replace(config.model, **changes))


def with_loss(config: RunConfig, **changes: Any) -> RunConfig:
    return replace(config, loss=replace(config.loss, **changes))


def without_groups(config: RunConfig, groups: Sequence[str]) -> RunConfig:
    return replace(config, features=tuple(g for g in config.features if g not in groups))


def dependents(group: str) -> tuple[str, ...]:
    """The groups that read group, directly or through another, in registry order."""
    found = {group}
    for _ in FEATURE_GROUPS:
        found |= {name for name, spec in FEATURE_GROUPS.items() if found & set(spec.requires)}
    return tuple(name for name in FEATURE_GROUPS if name in found - {group})


def drop_group(group: str) -> Variant:
    """The variant without a group and the groups that read it."""
    dropped = (group, *dependents(group))
    question = f"What does the model lose without {group}?"
    if len(dropped) == 2:
        question += f" {dropped[1]} reads it, so it goes too."
    elif len(dropped) > 2:
        question += f" {', '.join(dropped[1:-1])} and {dropped[-1]} read it, so they go too."
    return Variant(f"drop_{group}", question, lambda c: without_groups(c, dropped))


def unique_by_name(*groups: Sequence[Variant]) -> tuple[Variant, ...]:
    """The variants of groups in order, each name once."""
    return tuple({v.name: v for group in groups for v in group}.values())


BASELINE = Variant(BASELINE_VARIANT, "The built-in run", lambda c: c)
# One per built-in group but message_core, which the graph model needs.
DROPS = {g: drop_group(g) for g in BUILT_IN_GROUPS if g != "message_core"}
FEATURE_DROPS = tuple(DROPS.values())
CONTROLS = (
    Variant(
        "no_attention",
        "Does attention over sampled neighbours add anything beyond the root's own inputs, "
        "pool counts included?",
        lambda c: with_model(c, architecture="summary", slot_sum=False),
    ),
    Variant(
        "no_slot_sum",
        "Does the per-slot MLP sum help beyond attention?",
        lambda c: with_model(c, slot_sum=False),
    ),
    Variant(
        "no_pool_counts",
        "How much of the ranking comes from the candidate-pool counts?",
        lambda c: without_groups(c, POOL_GROUPS),
    ),
    Variant(
        "prior_weight",
        "Does the balanced positive weight beat textbook nnPU across seeds?",
        lambda c: with_loss(c, positive_weight="prior"),
    ),
    Variant(
        "no_weight_average",
        "Does selecting on the moving average of the weights help?",
        lambda c: with_training(c, weight_average_decay=0.0),
    ),
    DROPS["time_encoding"],
)
SUITES: dict[str, tuple[Variant, ...]] = {
    "controls": (BASELINE, *CONTROLS),
    "feature_drops": (BASELINE, *FEATURE_DROPS),
}
SUITES["all"] = unique_by_name(*SUITES.values())
VARIANTS = {variant.name: variant for variant in SUITES["all"]}
# The suite a run of the script without names runs.
DEFAULT_SUITE = "controls"


def select(names: Sequence[str] = ()) -> tuple[str, tuple[Variant, ...]]:
    """The name of the suite that names choose, and its variants, the baseline first.

    Each name is a suite or a variant; none chooses DEFAULT_SUITE. The baseline is always
    included and every variant runs once. The suite is named after the names given,
    joined by hyphens (results/experiments/<name>/).
    """
    chosen = tuple(dict.fromkeys(names or (DEFAULT_SUITE,)))
    unknown = [name for name in chosen if name not in SUITES and name not in VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown suites or variants {unknown}; suites: {list(SUITES)}, "
            f"variants: {list(VARIANTS)}"
        )
    groups = [SUITES[name] if name in SUITES else (VARIANTS[name],) for name in chosen]
    return "-".join(chosen), unique_by_name((BASELINE,), *groups)


def describe(base: RunConfig = DEFAULT_CONFIG) -> str:
    """The suites and variants, with each variant's question and changes, for --help."""
    width = max(map(len, [*SUITES, *VARIANTS])) + 2
    lines = [f"suites ({DEFAULT_SUITE} when none is named; the baseline is always included):"]
    for name, variants in SUITES.items():
        members = ", ".join(v.name for v in variants)
        lines += wrapped(name, members, width)
    seeds = ", ".join(map(str, SEEDS))
    lines += ["", f"variants (each trained with the seeds {seeds}):"]
    for variant in VARIANTS.values():
        lines += wrapped(variant.name, variant.question, width)
        lines += wrapped("", f"({variant.change_text(base)})", width)
    return "\n".join(lines)


def wrapped(name: str, text: str, width: int) -> list[str]:
    """A name and its text in two columns, the text wrapped at 88 characters."""
    body = textwrap.wrap(text, 88 - width - 2) or [""]
    return [f"  {(name if i == 0 else ''):<{width}}{line}" for i, line in enumerate(body)]
