"""Which configured world is loaded in TigerGraph, and the roles it plays (docs/design.md, "The
pipeline" and "Cache").

mule prepare runs read_world through the fetch executor, which refuses an account_type or
payment_rail outside the contract's value sets, and hashes it into the world's identity. It
then reads every identity file the loader wrote into the records folder that .env names, from
PhantomLedger's world record (world_<seed>.identity.toml), keeps the one whose shared fields equal
read_world's, comparing those fields only, and maps its seed to a configured world. The shared
fields are those of tigergraph.world.shared_fields, under the same key names; PhantomLedger
writes every one of them, zero counts included, so a file that lacks one, or holds another
under their names, is refused. read_world's degree sums are its own. Of the file's other keys
only the seed is read, to name the configured world, so no other key travels through data
code: not the commit, the clean-tree flag or the code-gate certificate.

A TOML key with dots is either a quoted key or a path of tables; both read here as the same
dotted name, and a file that holds one name both ways is refused.

Only one world is loaded in TigerGraph at a time, and mule train and mule evaluate never query it,
so mule prepare registers each world it prepares under its configured name (register): its identity
and its label identity, which name its cache and its examples. registered gives them back by name.
A world prepared again, such as one regenerated, replaces its entry, so a name always means the
world prepared last. The registry holds names and hashes, and no label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any
import uuid

from ..config import Rotation, WorldConfig, WorldsConfig
from ..paths import IDENTITY, pl_identity_files, world_registry
from ..tigergraph.executor import QueryExecutor
from ..tigergraph.queries import READ_WORLD, read_world
from ..tigergraph.queries.read_world import World
from ..tigergraph.world import SHARED_MAPS, SHARED_VALUES, identity, shared_fields

# The roles a world plays in a rotation.
TRAIN = "train"
SELECT = "select"
TEST = "test"
ROLES = (TRAIN, SELECT, TEST)
# The key of the identity file that names the world's seed.
SEED = "seed"


@dataclass(frozen=True)
class LoadedWorld:
    """The configured world loaded in TigerGraph: its settings, read_world's fields and their
    identity, the identity file that matched them, and the seed in it."""

    config: WorldConfig
    world: World
    identity: str
    path: Path
    seed: int


def load_world(executor: QueryExecutor) -> World:
    """read_world's fields, through the executor given (a fetch executor)."""
    return read_world.parse(executor.run(READ_WORLD, read_world.params()))


def _flatten(table: Mapping[str, Any], prefix: str, found: dict[str, object], path: Path) -> None:
    for key, value in table.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            nested: Mapping[str, Any] = value
            _flatten(nested, f"{name}.", found, path)
        elif name in found:
            raise ValueError(f"{path} holds the key {name} twice")
        else:
            found[name] = value


def read_record(path: Path) -> dict[str, object]:
    """An identity file's keys, each table's keys joined to its name with dots."""
    found: dict[str, object] = {}
    _flatten(tomllib.loads(path.read_text()), "", found, path)
    return found


def _shared(name: str) -> bool:
    return name in SHARED_VALUES or name.split(".", 1)[0] in SHARED_MAPS


def shared_record(
    record: Mapping[str, object], expected: Mapping[str, int], path: Path
) -> dict[str, int]:
    """The fields of an identity file that read_world shares, once the file holds exactly the
    keys expected under their names, each an integer."""
    found: dict[str, int] = {}
    for name, value in record.items():
        if not _shared(name):
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} holds {name} = {value!r}, not an integer")
        found[name] = value
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing or extra:
        raise ValueError(
            f"{path} does not hold read_world's shared fields: missing {missing}, unknown {extra}"
        )
    return dict(sorted(found.items()))


def match_world(world: World, files: Sequence[Path], worlds: WorldsConfig) -> LoadedWorld:
    """The configured world whose identity file's shared fields equal read_world's. Refused
    when no file matches, when two do, and when the matching file's seed is no configured
    world's."""
    expected = shared_fields(world)
    matched: list[tuple[Path, dict[str, object]]] = []
    for path in files:
        record = read_record(path)
        if shared_record(record, expected, path) == expected:
            matched.append((path, record))
    if not matched:
        raise ValueError(
            f"No identity file of {len(files)} matches the loaded world's read_world: load a "
            "world the loader verified, or check the records folder that .env names"
        )
    if len(matched) > 1:
        raise ValueError(
            f"The identity files {[str(path) for path, _ in matched]} all match the loaded "
            "world, so it is not known which world it is"
        )
    path, record = matched[0]
    seed = record.get(SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{path} holds no integer {SEED}, got {seed!r}")
    configured = [config for config in worlds.worlds if config.seed == seed]
    if not configured:
        raise ValueError(f"The loaded world, of seed {seed} ({path}), is no configured world")
    return LoadedWorld(configured[0], world, identity(world), path, seed)


def find_world(
    executor: QueryExecutor, worlds: WorldsConfig, files: Sequence[Path] | None = None
) -> LoadedWorld:
    """The configured world loaded in TigerGraph, matched among files, by default every
    identity file in the records folder that .env names."""
    return match_world(
        load_world(executor), pl_identity_files() if files is None else files, worlds
    )


@dataclass(frozen=True)
class Registered:
    """A world as mule prepare registered it: its configured name, the identity of the world it
    prepared, and the identity of the labels its examples were made from."""

    name: str
    identity: str
    labels: str


def _entries() -> dict[str, Registered]:
    """Every registered world, by name; none when no world has been prepared."""
    path = world_registry()
    if not path.is_file():
        return {}
    found = tomllib.loads(path.read_text()).get("world", {})
    if not isinstance(found, dict):
        raise ValueError(f"{path} must hold one [world.<name>] table per world")
    entries: dict[str, Registered] = {}
    for name, entry in found.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"identity", "labels"}
            or not all(
                isinstance(value, str) and IDENTITY.fullmatch(value) for value in entry.values()
            )
        ):
            raise ValueError(f"{path} holds {name} without an identity and a label identity")
        entries[str(name)] = Registered(str(name), entry["identity"], entry["labels"])
    return entries


def register(world: LoadedWorld, labels: str) -> None:
    """Register the loaded world under its configured name, replacing the world of that name
    prepared before, through a temporary file and a rename."""
    if not IDENTITY.fullmatch(labels):
        raise ValueError(f"A label identity is 32 hex digits, got {labels!r}")
    entries = _entries()
    entries[world.config.name] = Registered(world.config.name, world.identity, labels)
    text = "".join(
        f'[world.{name}]\nidentity = "{entry.identity}"\nlabels = "{entry.labels}"\n\n'
        for name, entry in sorted(entries.items())
    )
    path = world_registry()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def registered(name: str) -> Registered:
    """The world prepared under the configured name; refused when no world was."""
    found = _entries().get(name)
    if found is None:
        raise ValueError(
            f"No world is registered as {name}: load it in TigerGraph and run `mule prepare`"
        )
    return found


def roles(name: str, rotations: Sequence[Rotation]) -> frozenset[str]:
    """The roles the world of that name plays in the rotations: train, select and test."""
    found: set[str] = set()
    for rotation in rotations:
        found |= {role for role in ROLES if getattr(rotation, role) == name}
    return frozenset(found)
