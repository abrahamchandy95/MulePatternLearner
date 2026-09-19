"""Install and verify only the query definitions used by temporal training."""

import re
from typing import Any

from .dataset import QUERY_FILES, ROOT
from .source import TigerGraphExecutor

REVIEW_FILES = (
    *QUERY_FILES,
    "gsql/features/zelle_pair_time64.gsql",
    "gsql/features/payment_pair_time64.gsql",
)


def normalized(source: str) -> str:
    source = re.sub(r"/\*.*?\*/|//[^\n]*|#[^\n]*", "", source, flags=re.S)
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|[^\s"]+', source)
    return "".join(token if token.startswith('"') else token.lower() for token in tokens)


def definitions(source: str) -> dict[str, str]:
    starts = list(re.finditer(r"CREATE (?:OR REPLACE )?QUERY (\w+)", source, re.I))
    result = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(source)
        result[match[1]] = source[match.start() : end].split("USE GRAPH")[0].strip()
    return result


def verify_sources(executor: TigerGraphExecutor) -> list[str]:
    expected = {}
    for path in REVIEW_FILES:
        expected.update(definitions((ROOT / path).read_text()))
    for name, source in expected.items():
        live = str(executor.client.conn.gsql("USE GRAPH Mule_Pattern_Learner\nSHOW QUERY " + name))
        actual = definitions(live).get(name)
        if actual is None or normalized(actual) != normalized(source):
            raise ValueError("Installed query differs from repository source: " + name)
    return list(expected)


def verify_frozen_source(executor: TigerGraphExecutor, manifest: dict[str, Any]) -> None:
    """Recheck live provenance on every streamed run, including prepared-data reuse.

    Counts and headers catch drift, but cannot prove absence of same-count edits.
    The experiment still requires an operationally frozen source.
    """
    verify_sources(executor)
    conn = executor.client.conn
    if conn.getVertexCount("*", realtime=True) != manifest["source"]["source_counts"]:
        raise ValueError("Live graph counts changed; freeze the source and prepare a new dataset")
    config = manifest["config"]
    if config["evaluation_protocol"] == "strict_inductive":
        rows = conn.getVerticesById("Temporal_Training_Scope", [config["scope_id"]])
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("Prepared experiment scope is missing")
        attrs = rows[0]["attributes"]
        if (
            not attrs["ready"]
            or attrs["source_id"] != config["dataset_id"]
            or attrs["split_seed"] != int(config.get("split_seed", 42))
        ):
            raise ValueError("Prepared experiment scope is no longer valid")


def install(executor: TigerGraphExecutor) -> dict[str, str]:
    logs = {}
    schema = executor.client.conn.getSchema(force=True)
    if "Temporal_Training_Scope" not in {v["Name"] for v in schema["VertexTypes"]}:
        result = str(
            executor.client.conn.gsql(
                (ROOT / "gsql/schema/migrations/temporal_training_scope.gsql").read_text()
            )
        )
        if "Local schema change succeeded" not in result:
            raise RuntimeError(result)
        logs["scope_schema"] = result
    names = []
    for relative in REVIEW_FILES:
        source = (ROOT / relative).read_text()
        text = str(executor.client.conn.gsql(source))
        if "Successfully created queries" not in text or re.search(
            r"(?:[1-9]\d* syntax error|(?:Type Check|Semantic Check|Syntax) Error|draft query)",
            text,
            re.I,
        ):
            raise RuntimeError(text)
        logs[relative] = text
        names.extend(re.findall(r"CREATE OR REPLACE QUERY (\w+)", source))
    text = str(
        executor.client.conn.gsql(
            "USE GRAPH Mule_Pattern_Learner\nINSTALL QUERY " + ", ".join(names)
        )
    )
    if "Query installation finished" not in text or not re.search(r"failed: 0\b", text):
        raise RuntimeError(text)
    logs["install"] = text
    return logs
