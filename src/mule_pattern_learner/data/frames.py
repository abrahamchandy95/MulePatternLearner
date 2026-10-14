"""Parsed query rows as a frame."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import astuple, fields
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# Each field type of the parsed rows (tigergraph.queries.parse_row) and its column's type.
_DTYPES = {"str": "str", "int": "int64", "bool": "bool", "float": "float64"}


def frame(kind: type[DataclassInstance], rows: Sequence[DataclassInstance]) -> pd.DataFrame:
    """rows, each a kind, with one column per field of kind, of the field's type."""
    columns = fields(kind)
    return pd.DataFrame(
        [astuple(row) for row in rows], columns=[column.name for column in columns]
    ).astype({column.name: _DTYPES[str(column.type)] for column in columns})
