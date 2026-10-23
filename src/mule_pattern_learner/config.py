"""The settings as frozen dataclasses; DEFAULT_CONFIG holds every default.

RunConfig has one section per concern, and each component receives only its own
section. A section checks its values when it is built, so a bad setting fails before any
database work. `dataclasses.replace` changes a setting. Commands take no options: the worlds,
the roles, the dates and every other setting live here (docs/design.md, "Commands").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

# The ranges of the transport settings; the executor checks the ones it is given too.
QUERY_ATTEMPTS = range(1, 21)
OUTAGE_SECONDS = range(0, 86_401)
ACCOUNTS_PER_REQUEST = range(1, 1_025)
ACCOUNTS_PER_PART = range(1, 100_001)
# The response size above which TigerGraph refuses a query's answer, as the executor asks
# for it. Requests are sized to stay under half of it (docs/design.md, "Query plumbing").
RESPONSE_LIMIT_BYTES = 64_000_000
RESPONSE_BUDGETS = range(1, RESPONSE_LIMIT_BYTES // 2 + 1)
# The ranges of the calendar and the subgraph settings. A lookback must exceed the 30-day
# "new" window, or a start that is not known would count as new.
HISTORY_DAYS = range(1, 3_651)
SCORED_MONDAYS = range(1, 1_001)
LOOKBACK_DAYS = range(31, 3_651)
# The scored days of the export window, after its history: 52 Mondays fit in them
# (docs/design.md, "Worlds and calendar").
SCORED_DAYS = 365
CAPS = range(1, 1_025)
THRESHOLDS = range(1, 1_000_001)
# The current test reads 30 days back, the one value the design allows: no tenure built from
# uses is quiet for more than 21 days or ends more than 4 days after its last use's day, and
# 21 + 4 is under 30, so a link reads current from the past alone (docs/design.md, "Time").
# The queries write it as 2592000000, which the text rules require.
SHARING_DAYS = range(30, 31)
# The ranges of the worlds and the data settings.
SEEDS = range(0, 2**63)
PEOPLE = range(1, 1_000_000_001)
PER_POSITIVE = range(1, 1_001)
MIN_UNLABELLED = range(1, 10_000_001)
MIN_IDENTIFIED = range(1, 1_000_001)
FALLBACK_WEIGHTS = range(1, 1_001)
# blake2b takes a key of at most 64 bytes.
DRAW_KEY_BYTES = range(1, 65)
# The ranges of the model and training settings.
HIDDEN = range(1, 4_097)
HEADS = range(1, 65)
LAYERS = range(1, 9)
EPOCHS = range(1, 10_001)
BATCH_SIZES = range(1, 100_001)
# The ranges of the audit settings.
REPLICATES = range(1, 1_000_001)
SAMPLE_SIZES = range(1, 1_000_000_001)


def _integer(name: str, value: object, bounds: range) -> int:
    """value when it is an integer within bounds (a boolean is not one)."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in bounds:
        raise ValueError(
            f"{name} must be an integer in [{bounds.start}, {bounds[-1]}], got {value!r}"
        )
    return value


def _fraction(name: str, value: object, *, zero: bool) -> float:
    """value when it is a real number below 1 and above 0, or 0 itself when zero allows it
    (a boolean is not one)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not (0 <= value < 1 if zero else 0 < value < 1)
    ):
        interval = "[0, 1)" if zero else "(0, 1)"
        raise ValueError(f"{name} must be a number in {interval}, got {value!r}")
    return float(value)


def _between(name: str, value: object, low: float, high: float) -> float:
    """value when it is a real number strictly between low and high (a boolean is not one)."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not low < value < high:
        raise ValueError(f"{name} must be a number in ({low}, {high}), got {value!r}")
    return float(value)


def _positive(name: str, value: object) -> float:
    """value when it is a real number above 0 (a boolean is not one)."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ValueError(f"{name} must be a number above 0, got {value!r}")
    return float(value)


def _instance(name: str, value: object, kind: type) -> None:
    if not isinstance(value, kind):
        raise ValueError(f"{name} must be a {kind.__name__}, got {value!r}")


def _distinct(name: str, value: object) -> tuple[object, ...]:
    """value when it is a tuple without a repeated item."""
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple, got {value!r}")
    items: tuple[object, ...] = value
    if len(set(items)) != len(items):
        raise ValueError(f"{name} repeats an item: {value!r}")
    return items


def _date(name: str, value: object) -> date:
    """value when it is a date, not a datetime."""
    if type(value) is not date:
        raise ValueError(f"{name} must be a date, got {value!r}")
    return value


def _epoch_ms(day: date) -> int:
    """Milliseconds from 1970-01-01 00:00 UTC to day at 00:00 UTC."""
    return (day - date(1970, 1, 1)).days * 86_400_000


@dataclass(frozen=True)
class CalendarConfig:
    """The calendar every world shares (docs/design.md, "Worlds and calendar").

    The export starts on export_start (day 0, PhantomLedger's --start). Its first
    history_days days are history only; the scored Mondays follow, 00:00 UTC, from the first
    Monday at or after day history_days, inside the export's 365 scored days.
    """

    export_start: date = date(2024, 1, 1)
    history_days: int = 90
    scored_mondays: int = 52
    # Days before each Monday that the queries read.
    lookback_days: int = 90

    def __post_init__(self) -> None:
        start = _date("calendar.export_start", self.export_start)
        if start <= date(1970, 1, 1):
            raise ValueError(
                f"calendar.export_start must be after 1970-01-01, since every time is "
                f"unsigned milliseconds, got {start!r}"
            )
        _integer("calendar.history_days", self.history_days, HISTORY_DAYS)
        _integer("calendar.scored_mondays", self.scored_mondays, SCORED_MONDAYS)
        _integer("calendar.lookback_days", self.lookback_days, LOOKBACK_DAYS)
        if self.history_days < self.lookback_days:
            raise ValueError(
                f"calendar.history_days ({self.history_days}) must be at least "
                f"calendar.lookback_days ({self.lookback_days}), so every Monday has a full "
                "lookback inside the export"
            )
        if self.mondays()[-1] >= _epoch_ms(start + timedelta(days=self.export_days)):
            raise ValueError(
                f"calendar.scored_mondays ({self.scored_mondays}) do not fit in the "
                f"{SCORED_DAYS} scored days of the export"
            )

    @property
    def export_days(self) -> int:
        """The days of the export window: its history, then the scored days."""
        return self.history_days + SCORED_DAYS

    def mondays(self) -> tuple[int, ...]:
        """The scored Mondays, as UTC epoch milliseconds in increasing order."""
        first = self.export_start + timedelta(days=self.history_days)
        first += timedelta(days=-first.weekday() % 7)
        return tuple(
            _epoch_ms(first + timedelta(weeks=week)) for week in range(self.scored_mondays)
        )


# The kinds of world (docs/design.md, "Worlds and calendar"). Development worlds test the
# pipeline and tune the settings; reported worlds give the reported results; the test world
# is the 5,000-person world of the test on TigerGraph Cloud; a bench world only measures cost.
WORLD_KINDS = ("development", "reported", "test", "bench")
# The kinds of world a rotation is made of.
ROTATION_KINDS = ("development", "reported")


@dataclass(frozen=True)
class WorldConfig:
    """A PhantomLedger world: its name here, its fixed seed, its people and its kind."""

    name: str
    seed: int
    people: int
    kind: str

    def __post_init__(self) -> None:
        _instance("world.name", self.name, str)
        if not self.name.isidentifier():
            raise ValueError(
                f"world.name must be a name of letters, digits and _, got {self.name!r}"
            )
        _integer(f"world {self.name}.seed", self.seed, SEEDS)
        _integer(f"world {self.name}.people", self.people, PEOPLE)
        if self.kind not in WORLD_KINDS:
            raise ValueError(
                f"world {self.name}.kind must be one of {WORLD_KINDS}, got {self.kind!r}"
            )


@dataclass(frozen=True)
class Rotation:
    """The roles of three worlds, by name: one trains, one selects, one tests."""

    train: str
    select: str
    test: str

    def __post_init__(self) -> None:
        roles = (self.train, self.select, self.test)
        for role in roles:
            _instance("A rotation's world", role, str)
        if len(set(roles)) != 3:
            raise ValueError(f"A rotation needs three distinct worlds, got {roles!r}")


# The worlds of docs/design.md ("Worlds and calendar"): three reported worlds, three
# development worlds with other seeds, the 5,000-person test world and the 100,000-person
# world mule bench measures beside the development worlds and world A.
WORLDS = (
    WorldConfig("A", 2001, 300_000, "reported"),
    WorldConfig("B", 2002, 300_000, "reported"),
    WorldConfig("C", 2003, 300_000, "reported"),
    WorldConfig("D1", 1001, 30_000, "development"),
    WorldConfig("D2", 1002, 30_000, "development"),
    WorldConfig("D3", 1003, 30_000, "development"),
    WorldConfig("T", 3001, 5_000, "test"),
    WorldConfig("P100", 4001, 100_000, "bench"),
)
# Reported runs rotate the roles of the reported worlds; development uses one rotation.
ROTATIONS = (
    Rotation("A", "B", "C"),
    Rotation("B", "C", "A"),
    Rotation("C", "A", "B"),
    Rotation("D1", "D2", "D3"),
)


@dataclass(frozen=True)
class WorldsConfig:
    """The worlds and the rotations of their roles that the commands run."""

    worlds: tuple[WorldConfig, ...] = WORLDS
    rotations: tuple[Rotation, ...] = ROTATIONS

    def __post_init__(self) -> None:
        for world in _distinct("worlds.worlds", self.worlds):
            _instance("worlds.worlds item", world, WorldConfig)
        names = [world.name for world in self.worlds]
        seeds = [world.seed for world in self.worlds]
        if len(set(names)) != len(names) or len(set(seeds)) != len(seeds):
            raise ValueError("worlds.worlds must give every world its own name and seed")
        kinds = {world.name: world.kind for world in self.worlds}
        for rotation in _distinct("worlds.rotations", self.rotations):
            _instance("worlds.rotations item", rotation, Rotation)
        for rotation in self.rotations:
            roles = (rotation.train, rotation.select, rotation.test)
            unknown = [role for role in roles if role not in kinds]
            if unknown:
                raise ValueError(f"worlds.rotations names unknown worlds {unknown}")
            used = {kinds[role] for role in roles}
            if len(used) != 1 or not used <= set(ROTATION_KINDS):
                raise ValueError(
                    f"worlds.rotations {roles} must be three worlds of one kind of "
                    f"{ROTATION_KINDS}, got {sorted(used)}"
                )


@dataclass(frozen=True)
class TransportConfig:
    """The retry budgets and the size of TigerGraph requests. Changing them never changes
    results."""

    # Attempts of one request that count toward its retries (tigergraph.executor).
    max_query_attempts: int = 6
    # Wall-clock seconds one operation keeps retrying while TigerGraph is unavailable.
    max_outage_s: int = 900
    # Accounts of one Monday per per-account request; a refused request is halved.
    accounts_per_request: int = 32
    # The bytes a request's answer is sized to stay under: half of the response limit.
    response_budget_bytes: int = RESPONSE_LIMIT_BYTES // 2
    # Accounts of one Monday in a part of the cache, which is written once: a fetch that stops
    # loses at most the part it was filling.
    accounts_per_part: int = 1_024

    def __post_init__(self) -> None:
        _integer("transport.max_query_attempts", self.max_query_attempts, QUERY_ATTEMPTS)
        _integer("transport.max_outage_s", self.max_outage_s, OUTAGE_SECONDS)
        _integer("transport.accounts_per_request", self.accounts_per_request, ACCOUNTS_PER_REQUEST)
        _integer("transport.response_budget_bytes", self.response_budget_bytes, RESPONSE_BUDGETS)
        _integer("transport.accounts_per_part", self.accounts_per_part, ACCOUNTS_PER_PART)


@dataclass(frozen=True)
class SubgraphConfig:
    """The caps, thresholds and windows of the queries (docs/design.md, "Time" and "Shared
    rules"). Every value is the design's, tuned and frozen on development worlds only; the
    same settings build every subgraph."""

    # Days before t that every query reads.
    lookback_days: int = 90
    # Payments per time-ordered list (B1's inflows and outflows, B2's new payers' first
    # payments, B3's payments per direction).
    recent_payments: int = 256
    # Payments per direction of the scored account, and of each expanded counterparty (B4).
    scored_payments: int = 32
    counterparty_payments: int = 16
    # Other accounts of the scored account's customer (B4).
    own_accounts: int = 8
    # Devices and IPs of each kind per kind of link, per customer and per account (B5).
    devices_ips: int = 8
    # Co-user candidates per scored account (B5), and community members (B6).
    co_users: int = 48
    community: int = 64
    # Distinct customers above which an account, a device or an IP is a hub, and a device
    # or an IP a crowd.
    hub_threshold: int = 200
    crowd_threshold: int = 8
    # The windows over which a link, a pair or a customer is new, in days.
    new_windows: tuple[int, ...] = (7, 30)
    # Days in which a link held, or a use on a payment, makes a customer share a node: fixed
    # at 30 (SHARING_DAYS).
    sharing_days: int = 30

    def __post_init__(self) -> None:
        lookback = _integer("subgraph.lookback_days", self.lookback_days, LOOKBACK_DAYS)
        for name in (
            "recent_payments",
            "scored_payments",
            "counterparty_payments",
            "own_accounts",
            "devices_ips",
            "co_users",
            "community",
        ):
            _integer(f"subgraph.{name}", getattr(self, name), CAPS)
        hub = _integer("subgraph.hub_threshold", self.hub_threshold, THRESHOLDS)
        crowd = _integer("subgraph.crowd_threshold", self.crowd_threshold, THRESHOLDS)
        if crowd >= hub:
            raise ValueError(
                f"subgraph.crowd_threshold ({crowd}) must be below subgraph.hub_threshold ({hub})"
            )
        windows = _distinct("subgraph.new_windows", self.new_windows)
        if not windows:
            raise ValueError("subgraph.new_windows must hold a window")
        for window in windows:
            _integer("subgraph.new_windows", window, range(1, lookback))
        if list(windows) != sorted(self.new_windows):
            raise ValueError(f"subgraph.new_windows must increase, got {self.new_windows!r}")
        _integer("subgraph.sharing_days", self.sharing_days, SHARING_DAYS)


# How the selection and test worlds are scored (docs/design.md, "If whole-world scoring is
# too slow"): every scorable account on every Monday, or the fallback sample.
SCORING_MODES = ("whole", "fallback")


@dataclass(frozen=True)
class DataConfig:
    """The training examples, the keyed draws and the scoring mode (docs/design.md,
    "Worlds, examples and weights")."""

    # Unlabelled accounts drawn on a training Monday per positive, and at least this many.
    per_positive: int = 15
    min_unlabelled: int = 500
    # The key of the blake2b draws, pinned by a golden test.
    draw_key: bytes = b"mule-draw-v1\x00\x00\x00\x00"
    # mule prepare refuses a selection world with fewer identified active mules.
    min_selection_identified: int = 5
    scoring: str = "whole"
    # Under the fallback, each sampled account stands for this many: its inclusion chance
    # is the inverse.
    fallback_weight: int = 10

    def __post_init__(self) -> None:
        _integer("data.per_positive", self.per_positive, PER_POSITIVE)
        _integer("data.min_unlabelled", self.min_unlabelled, MIN_UNLABELLED)
        _instance("data.draw_key", self.draw_key, bytes)
        if len(self.draw_key) not in DRAW_KEY_BYTES:
            raise ValueError(f"data.draw_key must be 1 to 64 bytes, got {self.draw_key!r}")
        _integer("data.min_selection_identified", self.min_selection_identified, MIN_IDENTIFIED)
        if self.scoring not in SCORING_MODES:
            raise ValueError(f"data.scoring must be one of {SCORING_MODES}, got {self.scoring!r}")
        _integer("data.fallback_weight", self.fallback_weight, FALLBACK_WEIGHTS)


@dataclass(frozen=True)
class ModelConfig:
    """The heterogeneous attention model (docs/design.md, "The model"); every value is the
    design's."""

    # The width of every node's representation after each layer, split among the heads.
    hidden: int = 64
    heads: int = 4
    # The share of each node's representation zeroed after each layer while training.
    dropout: float = 0.2
    # Message-passing layers, and the head's hidden width.
    layers: int = 2
    head_hidden: int = 32
    # Whether the head reads the community block; the ablation without it does not.
    community: bool = True

    def __post_init__(self) -> None:
        _integer("model.hidden", self.hidden, HIDDEN)
        _integer("model.heads", self.heads, HEADS)
        if self.hidden % self.heads:
            raise ValueError(f"model.hidden ({self.hidden}) must split evenly among model.heads")
        _fraction("model.dropout", self.dropout, zero=True)
        _integer("model.layers", self.layers, LAYERS)
        _integer("model.head_hidden", self.head_hidden, HIDDEN)
        _instance("model.community", self.community, bool)


@dataclass(frozen=True)
class TrainingConfig:
    """The optimiser, the epochs, the seeds and the prior grid (docs/design.md, "Training
    and selection"); every value is the design's."""

    # Adam's learning rate and weight decay.
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    # Examples per optimiser step.
    batch_size: int = 128
    # Epochs at most per prior, and epochs without a better selection score before it stops.
    max_epochs: int = 30
    patience: int = 5
    # The model seeds: each varies only the initial weights, dropout and batch order.
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    # The seed of the one shuffle of a world's training examples into batches, the same for
    # every model seed, and the seed of the draws of the paired intervals that decide the
    # prior grid. Both stay fixed, since the three worlds vary the data.
    shuffle_seed: int = 0
    bootstrap_seed: int = 0
    # The nnPU priors tried, as multiples of the identified active share of the training
    # world.
    prior_multiples: tuple[float, ...] = (1, 3, 10)
    # None runs the grid; a multiple fixes the prior at it, as the design fixes it at 1 for
    # reported runs when no pair of the grid's priors differs on the development rotation.
    fixed_prior: float | None = None

    def __post_init__(self) -> None:
        _fraction("training.learning_rate", self.learning_rate, zero=False)
        _fraction("training.weight_decay", self.weight_decay, zero=True)
        _integer("training.batch_size", self.batch_size, BATCH_SIZES)
        _integer("training.max_epochs", self.max_epochs, EPOCHS)
        _integer("training.patience", self.patience, EPOCHS)
        if not _distinct("training.seeds", self.seeds):
            raise ValueError("training.seeds must hold a seed")
        for seed in self.seeds:
            _integer("training.seeds", seed, SEEDS)
        _integer("training.shuffle_seed", self.shuffle_seed, SEEDS)
        _integer("training.bootstrap_seed", self.bootstrap_seed, SEEDS)
        if not _distinct("training.prior_multiples", self.prior_multiples):
            raise ValueError("training.prior_multiples must hold a multiple")
        for multiple in self.prior_multiples:
            _positive("training.prior_multiples", multiple)
        if self.fixed_prior is not None:
            _positive("training.fixed_prior", self.fixed_prior)


@dataclass(frozen=True)
class AuditConfig:
    """The bounds of mule audit's checks and of the evaluation reports (docs/design.md,
    "Leak rules and how tests enforce them" and "Evaluation"); every value is the design's,
    the owner's of 2026-10-17 among them."""

    # The clock probe passes when the upper end of its ROC AUC interval is below this.
    clock_probe_upper: float = 0.55
    # A feature whose ROC AUC is at least this, or at most 1 minus it, stops the run.
    separation_auc: float = 0.95
    # A near-pure region: a flag value, or the top or bottom near_pure_tail of scored
    # account-Mondays, holding at least near_pure_min_mule_share of active true-mule
    # account-Mondays, in which true mules are more than near_pure_purity.
    near_pure_tail: float = 0.001
    near_pure_min_mule_share: float = 0.01
    near_pure_purity: float = 0.5
    # The top share of a Monday's ranking a hidden mule must reach not to be missed or weak.
    report_budget: float = 0.05
    # Ring-resampled replicates, and the level of every interval.
    replicates: int = 1000
    interval: float = 0.95
    # Non-mule account-Mondays the clock probe samples.
    nonmule_sample: int = 200_000
    # The level of each onset month's binomial band in the identification horizon check: a
    # 1% chance over the 12 months that a month falls outside by chance (Bonferroni).
    horizon_level: float = 1 - 0.01 / 12

    def __post_init__(self) -> None:
        _between("audit.clock_probe_upper", self.clock_probe_upper, 0.5, 1)
        _between("audit.separation_auc", self.separation_auc, 0.5, 1)
        _between("audit.near_pure_tail", self.near_pure_tail, 0, 0.5)
        _fraction("audit.near_pure_min_mule_share", self.near_pure_min_mule_share, zero=False)
        _fraction("audit.near_pure_purity", self.near_pure_purity, zero=False)
        _fraction("audit.report_budget", self.report_budget, zero=False)
        _integer("audit.replicates", self.replicates, REPLICATES)
        _fraction("audit.interval", self.interval, zero=False)
        _integer("audit.nonmule_sample", self.nonmule_sample, SAMPLE_SIZES)
        _fraction("audit.horizon_level", self.horizon_level, zero=False)


@dataclass(frozen=True)
class ScoreConfig:
    """The accounts and Mondays mule score scores (docs/design.md, "Commands")."""

    accounts: tuple[str, ...] = ()
    dates: tuple[date, ...] = ()

    def __post_init__(self) -> None:
        for account in _distinct("score.accounts", self.accounts):
            if not isinstance(account, str) or not account:
                raise ValueError(f"score.accounts must be account ids, got {account!r}")
        for day in _distinct("score.dates", self.dates):
            if _date("score.dates", day).weekday() != 0:
                raise ValueError(f"score.dates must be Mondays, got {day!r}")


@dataclass(frozen=True)
class RunConfig:
    """Every setting; the defaults are DEFAULT_CONFIG."""

    calendar: CalendarConfig = CalendarConfig()
    worlds: WorldsConfig = WorldsConfig()
    transport: TransportConfig = TransportConfig()
    subgraph: SubgraphConfig = SubgraphConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    audit: AuditConfig = AuditConfig()
    score: ScoreConfig = ScoreConfig()

    def __post_init__(self) -> None:
        _instance("calendar", self.calendar, CalendarConfig)
        _instance("worlds", self.worlds, WorldsConfig)
        _instance("transport", self.transport, TransportConfig)
        _instance("subgraph", self.subgraph, SubgraphConfig)
        _instance("data", self.data, DataConfig)
        _instance("model", self.model, ModelConfig)
        _instance("training", self.training, TrainingConfig)
        _instance("audit", self.audit, AuditConfig)
        _instance("score", self.score, ScoreConfig)
        if self.calendar.lookback_days != self.subgraph.lookback_days:
            raise ValueError(
                f"calendar.lookback_days ({self.calendar.lookback_days}) must equal "
                f"subgraph.lookback_days ({self.subgraph.lookback_days})"
            )


DEFAULT_CONFIG = RunConfig()
