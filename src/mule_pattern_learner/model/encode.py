"""The layout of every row a batch carries, before and after encoding (docs/design.md,
"Encoding", and docs/features.md, "Widths").

A built batch keeps each value as the cache gives it: counts and integer cents as they are,
shares in [0, 1], flags 0 or 1, and each duration as a presence bit and integer milliseconds.
A Layout names a row's columns in order, each with its kind, so that one row of numbers can be
checked before it is used and read by name in a test.

Every raw row is a float64 array: a share is a fraction, and every integer a row holds (counts,
cents and milliseconds within the 90-day lookback) is below 2**53, where float64 holds each
integer exactly. check refuses one that is not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from .time_encoding import WIDTH

type EdgeType = tuple[str, str, str]

# The largest integer a raw row may hold: float64 holds every integer up to it exactly.
EXACT = 2**53


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
    reverse), the layout of each relation that carries attributes, and the community block's.
    A relation without a layout, such as ownership, carries none."""

    nodes: Mapping[str, Layout]
    relations: tuple[EdgeType, ...]
    edges: Mapping[EdgeType, Layout]
    community: Layout

    def __post_init__(self) -> None:
        if len(set(self.relations)) != len(self.relations):
            raise ValueError("A batch layout names a relation twice")
        for source, _, target in self.relations:
            if source not in self.nodes or target not in self.nodes:
                raise ValueError(f"A relation joins a node type without a layout: {source}")
        if not set(self.edges) <= set(self.relations):
            raise ValueError("A batch layout gives attributes to a relation it does not hold")
