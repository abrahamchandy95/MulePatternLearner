"""Context scopes and the Account loading columns against the canonical DDL."""

from __future__ import annotations

import re

import pytest

from mule_pattern_learner.contract import graph_schema
from mule_pattern_learner.contract.graph_schema import context_scope
from mule_pattern_learner.paths import GSQL_DIR


def test_strict_claim_fails_before_preparation_or_training() -> None:
    for missing in (None, ""):
        with pytest.raises(ValueError, match="frozen TigerGraph scope id"):
            context_scope(missing)
    assert context_scope("scope") == "scope"


def test_account_schema_contract_matches_canonical_ddl() -> None:
    ddl = (GSQL_DIR / "schema/schema.gsql").read_text()
    block = ddl.split("ADD VERTEX Account (", 1)[1].split(") WITH", 1)[0]
    fields = re.findall(
        r"^\s*(?:PRIMARY_ID )?(\w+)\s+(?:STRING|BOOL|UINT|INT)", block, re.MULTILINE
    )
    assert fields == graph_schema.ACCOUNT_STORAGE_COLUMNS
    assert re.search(r"is_mule INT DEFAULT 0", block)
    loader = (GSQL_DIR / "schema/account_loading.gsql").read_text()
    columns = re.findall(r'\$"(\w+)"', loader)
    assert columns == graph_schema.ACCOUNT_STORAGE_COLUMNS
    header = loader.split("DEFINE HEADER account_header =", 1)[1].split(";", 1)[0]
    assert re.findall(r'"(\w+)"', header) == graph_schema.ACCOUNT_LOAD_COLUMNS
