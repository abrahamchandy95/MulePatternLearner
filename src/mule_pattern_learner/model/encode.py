"""The layout of every row a batch carries, and its encoding into the model's inputs
(docs/design.md, "Encoding", and docs/features.md, "Widths").

A built batch keeps each value as the cache gives it: counts and integer cents as they are,
shares in [0, 1], flags 0 or 1, and each duration as a presence bit and integer milliseconds.
A Layout names a row's columns in order, each with its kind, so that one row of numbers can be
checked before it is used and read by name in a test.

Every raw row is a float64 array: a share is a fraction, and every integer a row holds (counts,
cents and milliseconds within the 90-day lookback) is below 2**53, where float64 holds each
integer exactly. check refuses one that is not.

encode turns raw rows into the model's float32 inputs on the device that trains, each step:
a count n as log(1 + n), cents as log(1 + dollars), shares and flags as they are, and each
duration as its presence bit and the 64 Fourier numbers of model.time_encoding, all 0 when it
is absent. Every column keeps its place, a duration's 65 numbers in the place of its two raw
columns. The milliseconds reach the Fourier basis as integers, read from float64 exactly, so a
batch kept on a device without float64 is encoded from the host.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from . import time_encoding
from .time_encoding import WIDTH

type EdgeType = tuple[str, str, str]

# The largest integer a raw row may hold: float64 holds every integer up to it exactly.
EXACT = 2**53
# Cents in a dollar: an amount is encoded as log(1 + dollars).
CENTS = 100.0


class Kind(StrEnum):
    """What a column holds, and so how it is encoded (docs/design.md, "Encoding")."""

    # 0 or 1, as it is; a one-hot rail is seven of them.
    FLAG = "flag"
    # A count n, encoded as log(1 + n).
    COUNT = "count"
    # Integer cents, encoded as log(1 + dollars).
    AMOUNT = "amount"
    # A share in [0, 1], as it is.
    SHARE = "share"
    # A duration: a presence bit and integer milliseconds raw, 0 when absent; encoded as the
    # presence bit and the 64 Fourier numbers of model.time_encoding, all 0 when absent.
    DURATION = "duration"


@dataclass(frozen=True)
class Column:
    """A column of a row: its name, as docs/features.md names the feature, and its kind."""

    name: str
    kind: Kind

    @property
    def raw_names(self) -> tuple[str, ...]:
        """The names of its raw columns: a duration's presence bit and its milliseconds."""
        if self.kind is Kind.DURATION:
            return (f"{self.name}_present", f"{self.name}_ms")
        return (self.name,)

    @property
    def width(self) -> int:
        """Its width once encoded."""
        return 1 + WIDTH if self.kind is Kind.DURATION else 1


@dataclass
class _Plan:
    """Where a layout's raw columns go once encoded: the raw and the encoded position of each
    column that is not a duration, whether it is logged and what it is divided by first; the
    raw positions of each duration's presence bit and milliseconds, the encoded position of
    its presence bit, and the encoded positions of its 64 Fourier numbers, duration by
    duration."""

    plain_raw: list[int] = field(default_factory=list[int])
    plain_out: list[int] = field(default_factory=list[int])
    logged: list[bool] = field(default_factory=list[bool])
    divisor: list[float] = field(default_factory=list[float])
    present_raw: list[int] = field(default_factory=list[int])
    ms_raw: list[int] = field(default_factory=list[int])
    present_out: list[int] = field(default_factory=list[int])
    fourier_out: list[int] = field(default_factory=list[int])


@dataclass(frozen=True)
class Layout:
    """The columns of one kind of row, in the order of docs/features.md's table."""

    columns: tuple[Column, ...]

    def __post_init__(self) -> None:
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"A layout names a column twice: {names}")

    @cached_property
    def raw_names(self) -> tuple[str, ...]:
        """The names of the raw columns, in order."""
        return tuple(name for column in self.columns for name in column.raw_names)

    @property
    def raw_width(self) -> int:
        return len(self.raw_names)

    @property
    def width(self) -> int:
        """The width once encoded."""
        return sum(column.width for column in self.columns)

    def index(self, name: str) -> int:
        """The position of a raw column, by name."""
        return self.raw_names.index(name)

    @cached_property
    def plan(self) -> _Plan:
        """Where each raw column goes once encoded."""
        plan = _Plan()
        raw = out = 0
        for column in self.columns:
            if column.kind is Kind.DURATION:
                plan.present_raw.append(raw)
                plan.ms_raw.append(raw + 1)
                plan.present_out.append(out)
                plan.fourier_out += range(out + 1, out + 1 + WIDTH)
            else:
                plan.plain_raw.append(raw)
                plan.plain_out.append(out)
                plan.logged.append(column.kind in (Kind.COUNT, Kind.AMOUNT))
                plan.divisor.append(CENTS if column.kind is Kind.AMOUNT else 1.0)
            raw += len(column.raw_names)
            out += column.width
        return plan

    def positions(self, *names: str) -> list[int]:
        """The raw positions of the columns named, a duration's presence bit and milliseconds
        both, in the layout's order."""
        unknown = set(names) - {column.name for column in self.columns}
        if unknown:
            raise KeyError(f"The layout has no column {sorted(unknown)}")
        found: list[int] = []
        position = 0
        for column in self.columns:
            width = len(column.raw_names)
            if column.name in names:
                found += range(position, position + width)
            position += width
        return found

    def check(self, raw: NDArray[np.float64], what: str) -> None:
        """Refuse raw rows of this layout that hold a value its kind cannot: a flag or a
        presence bit other than 0 or 1, a count, cents or milliseconds that is not a whole
        number in [0, 2**53), a share outside [0, 1], an absent duration that is not 0, or
        anything not finite. what names the rows in the error."""
        if raw.ndim != 2 or raw.shape[1] != self.raw_width:
            raise ValueError(
                f"The rows of {what} have the shape {raw.shape}, not (n, {self.raw_width})"
            )
        if not np.isfinite(raw).all():
            raise ValueError(f"The rows of {what} hold a value that is not finite")
        position = 0
        for column in self.columns:
            values = raw[:, position]
            if column.kind is Kind.DURATION:
                present, ms = values, raw[:, position + 1]
                _flags(present, f"{what}.{column.name}_present")
                _whole(ms, f"{what}.{column.name}_ms")
                if np.any((present == 0) & (ms != 0)):
                    raise ValueError(f"{what}.{column.name} is absent but not 0")
            elif column.kind is Kind.FLAG:
                _flags(values, f"{what}.{column.name}")
            elif column.kind is Kind.SHARE:
                if np.any((values < 0) | (values > 1)):
                    raise ValueError(f"{what}.{column.name} holds a share outside [0, 1]")
            else:
                _whole(values, f"{what}.{column.name}")
            position += len(column.raw_names)


def _flags(values: NDArray[np.float64], what: str) -> None:
    if np.any((values != 0) & (values != 1)):
        raise ValueError(f"{what} holds a value other than 0 or 1")


def _whole(values: NDArray[np.float64], what: str) -> None:
    if np.any((values < 0) | (values >= EXACT) | (values != np.floor(values))):
        raise ValueError(f"{what} holds a value that is not a whole number in [0, 2**53)")


@dataclass(frozen=True)
class BatchLayout:
    """The layouts of a batch: each node type's, every relation (each forward relation and its
    reverse), the layout of each relation that carries attributes, the community block's, and
    the node type of the scored accounts. A relation without a layout, such as ownership,
    carries none."""

    nodes: Mapping[str, Layout]
    relations: tuple[EdgeType, ...]
    edges: Mapping[EdgeType, Layout]
    community: Layout
    scored: str

    def __post_init__(self) -> None:
        if self.scored not in self.nodes:
            raise ValueError(f"The scored node type {self.scored!r} has no layout")
        if len(set(self.relations)) != len(self.relations):
            raise ValueError("A batch layout names a relation twice")
        for source, _, target in self.relations:
            if source not in self.nodes or target not in self.nodes:
                raise ValueError(f"A relation joins a node type without a layout: {source}")
        if not set(self.edges) <= set(self.relations):
            raise ValueError("A batch layout gives attributes to a relation it does not hold")


def encode(raw: Tensor, layout: Layout, device: torch.device | str | None = None) -> Tensor:
    """layout's raw rows encoded on device, raw's own when None, as float32: counts as
    log(1 + n), cents as log(1 + dollars), shares and flags as they are, and each duration as
    its presence bit and its 64 Fourier numbers, all 0 when it is absent."""
    if raw.dim() != 2 or raw.shape[1] != layout.raw_width:
        raise ValueError(f"Raw rows have the shape {tuple(raw.shape)}, not (n, {layout.raw_width})")
    if raw.dtype != torch.float64:
        raise TypeError(f"Raw rows are float64, not {raw.dtype}")
    target = raw.device if device is None else torch.device(device)
    plan = layout.plan
    found = torch.empty((raw.shape[0], layout.width), dtype=torch.float32, device=target)
    plain = raw[:, plan.plain_raw]
    logged = torch.tensor(plan.logged, dtype=torch.bool, device=raw.device)
    divisor = torch.tensor(plan.divisor, dtype=raw.dtype, device=raw.device)
    plain = torch.where(logged, torch.log1p(plain / divisor), plain)
    found[:, plan.plain_out] = plain.to(device=target, dtype=torch.float32)
    present = raw[:, plan.present_raw].to(device=target, dtype=torch.float32)
    milliseconds = raw[:, plan.ms_raw].to(torch.int64).to(target)
    fourier = time_encoding.encode(milliseconds) * present[..., None]
    found[:, plan.present_out] = present
    found[:, plan.fourier_out] = fourier.reshape(raw.shape[0], len(plan.fourier_out))
    return found


def encode_batch(
    data: HeteroData, layout: BatchLayout, device: torch.device | str | None = None
) -> HeteroData:
    """A raw batch encoded on device, the batch's own when None: every node row, every
    relation's edges with its attributes, scored_index, the community block, and in a
    training batch label and weight, the weight as float32. A reverse relation's attributes
    are its forward relation's, encoded once."""
    if set(data.node_types) != set(layout.nodes) or set(data.edge_types) != set(layout.relations):
        raise ValueError("The batch's node and edge types are not the layout's")
    scored: Tensor = data.scored_index
    target = scored.device if device is None else torch.device(device)
    found = HeteroData()
    for kind, rows in layout.nodes.items():
        found[kind].x = encode(data[kind].x, rows, target)
    encoded: dict[int, Tensor] = {}
    for relation in layout.relations:
        store = data[relation]
        found[relation].edge_index = store.edge_index.to(target)
        attributes = layout.edges.get(relation)
        if attributes is None:
            if "edge_attr" in store:
                raise ValueError(f"The relation {relation[1]} carries attributes it has none of")
            continue
        attrs: Tensor = store.edge_attr
        if id(attrs) not in encoded:
            encoded[id(attrs)] = encode(attrs, attributes, target)
        found[relation].edge_attr = encoded[id(attrs)]
    found.scored_index = scored.to(target)
    found.community = encode(data.community, layout.community, target)
    if "label" in data:
        found.label = data.label.to(target)
        found.weight = data.weight.to(device=target, dtype=torch.float32)
    return found
