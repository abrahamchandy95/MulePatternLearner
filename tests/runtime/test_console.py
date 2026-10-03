"""The console: each event's short line, the progress rewritten in place, and nothing else."""

from __future__ import annotations

import ast
import io
from pathlib import Path
import sys
from typing import Any

import pytest

from mule_pattern_learner.paths import REPOSITORY_ROOT
from mule_pattern_learner.runtime import console
from mule_pattern_learner.runtime.console import (
    LINES,
    event_text,
    show,
    show_event,
    show_scoring,
)


class Terminal(io.StringIO):
    """A stdout that says it is a terminal."""

    def isatty(self) -> bool:
        return True


# Records as the commands emit them; most are the owner's first run on the CUDA host.
KNOWN = {"train": 20, "validation": 11, "test": 20}
STALE = [f"query_{n}" for n in range(12)]


def records(run: Path) -> list[tuple[dict[str, Any], str | None]]:
    """(record, its console text) pairs; ``run`` is a run directory under the cwd."""
    plan = {"epochs": 30, "steps_per_epoch": 100, "patience": 6, "stopped": False}
    host = {"device": "cuda", "threads": 4, "deterministic": True, "sampler_backend": "cugraph"}
    epoch = {"loss": 0.0991, "steps": 100, "weights": "averaged", "validation_roc_auc": 0.9592}
    return [
        (
            {
                "event": "retry",
                "failure": "availability",
                "operation": "connect",
                "attempt": 2,
                "retry_in_s": 5.8,
                "error": "TigerGraphException: Cannot parse json: <html> <title>Starting "
                "workspace</title> ...",
                "reason": "starting workspace, HTTP 502",
            },
            "TigerGraph is not answering yet (starting workspace, HTTP 502): attempt 2, "
            "retrying in 6 s",
        ),
        (
            {
                "event": "retry",
                "failure": "server_timeout",
                "operation": "fetch_training_context (64 keys)",
                "attempt": 1,
                "retry_in_s": 3.2,
                "error": "TigerGraphException: Query timeout exceeded",
                "reason": "query timeout exceeded",
            },
            "TigerGraph timed out on fetch_training_context, 64 keys (query timeout "
            "exceeded): attempt 1, retrying in 3 s",
        ),
        (
            {
                "event": "retry",
                "failure": "deterministic",
                "operation": "CREATE QUERY queries/training_context.gsql",
                "attempt": 1,
                "retry_in_s": 4.4,
                "error": "TigerGraphException: Runtime Error: out of memory",
                "reason": "runtime error: out of memory",
            },
            "TigerGraph failed CREATE QUERY queries/training_context.gsql (runtime error: out "
            "of memory): attempt 1, retrying in 4 s",
        ),
        (
            {"event": "context_split", "keys": 64, "hop": 1, "error": "ServerTimeoutError: x"},
            "TigerGraph timed out on a request of 64 keys at hop 1; requesting its halves on "
            "their own",
        ),
        (
            {"event": "install", "stale": STALE, "up_to_date": []},
            "Installing all 12 queries on TigerGraph (about 50 minutes)...",
        ),
        # Only an install of every query gives the time a fresh graph's install took.
        (
            {"event": "install", "stale": STALE[:3], "up_to_date": STALE[3:]},
            "Installing 3 queries on TigerGraph...",
        ),
        ({"event": "install", "stale": [], "up_to_date": STALE}, None),
        (
            {
                "event": "install_unanswered",
                "error": "ServerTimeoutError",
                "note": "polling the endpoint listing",
            },
            "TigerGraph has not answered the install request (ServerTimeoutError); waiting "
            "until the installed queries are enabled",
        ),
        (
            {"event": "install_wait", "awaiting": STALE[:2], "elapsed_s": 840},
            "Waiting for 2 queries to compile: 14.0 min so far",
        ),
        (
            {"event": "installed", "installed": STALE, "seconds": 2880},
            "Installed 12 queries in 48.0 min",
        ),
        (
            {"event": "drop_retired", "dropped": ["old_a", "old_b"]},
            "Dropped 2 retired queries: old_a, old_b",
        ),
        ({"event": "drop_retired", "dropped": []}, None),
        (
            {"event": "scope", "scope": "strict_mule_v2", "unowned_members": {"gl": 9}},
            "Scope strict_mule_v2 is in place",
        ),
        (
            {"event": "scope", "scope": "strict_mule_v2", "creating": True},
            "Creating scope strict_mule_v2 on TigerGraph...",
        ),
        ({"event": "scope", "scope": "strict_mule_v2", "created": True}, None),
        (
            {
                "event": "reveal",
                "labels": "already revealed",
                "known_labels": 752623,
                "revealed_labels": 51,
                "contract": {"known_labels": 752623},
            },
            "Known mules already revealed on TigerGraph: 51",
        ),
        (
            {"event": "hubs", "hub_counts": {"101": {"1": 3, "2": 1}, "102": {"1": 2, "2": 0}}},
            "Hub registry: 6 hub rows over 2 cutoffs",
        ),
        (
            {
                "event": "dataset",
                "dataset_id": "1a2b3c4d5e6f7a8b9c",
                "status": "reused",
                "known_mules": KNOWN,
            },
            "Dataset 1a2b3c4d5e6f: 20 / 11 / 20 known mules in train / validation / test",
        ),
        (
            {"event": "already_complete", "run": str(run)},
            "results/baseline/seed-42 is complete; its metrics.json is summarised below",
        ),
        (
            {"event": "start", **host, "known_mules": KNOWN, "run": str(run), **plan}
            | {"epoch": 0, "step": 0, "database_calls": 0, "elapsed_seconds": 0.1},
            "Training on cuda (cuGraph sampler) into results/baseline/seed-42: 100 steps per "
            "epoch, at most 30 epochs, early stop after 6 epochs without gain",
        ),
        (
            {"event": "resume", **host, "run": str(run), **plan, "epoch": 2, "step": 40}
            | {"sampler_backend": "torch", "device": "cpu", "patience": 0},
            "Resuming results/baseline/seed-42 at epoch 3, step 40, on cpu (torch sampler): "
            "100 steps per epoch, at most 30 epochs, no early stop",
        ),
        (
            {
                "event": "train",
                "epoch": 3,
                "step": 60,
                "steps": 100,
                "date": "2024-07-01",
                "loss": 0.13612,
                "objective": 0.13612,
                "seconds_per_step": 1.94,
                "batch": {"roots": 64},
                "database_calls": 1234,
            },
            "epoch 3  step 60/100  loss 0.136  1.9 s/step",
        ),
        (
            {"event": "score", "split": "validation", "accounts": 640, "total": 2011},
            None,
        ),
        (
            {"event": "epoch", "epoch": 5, **epoch, "validation_ap": 0.5661}
            | {"selected": True, "stopped": False, "best_epoch": 5, "epoch_seconds": 186.0},
            "epoch  5  loss 0.099  validation AP 0.566  ROC AUC 0.959  3.1 min  best so far",
        ),
        (
            {"event": "epoch", "epoch": 11, **epoch, "validation_ap": None}
            | {"selected": False, "stopped": True, "best_epoch": 5, "epoch_seconds": 41.0},
            "epoch 11  loss 0.099  validation AP n/a    ROC AUC 0.959     41 s\n"
            "early stop: no gain for 6 epochs",
        ),
        ({"event": "complete", "best_epoch": 5}, None),
        (
            {"event": "sampler_backend", "saved": "cugraph", "resumed": "torch", "note": "x"},
            "Note: the run sampled with the cugraph backend and resumes with torch, so the "
            "remaining steps sample a different stream",
        ),
        (
            {
                "event": "host_settings",
                "saved": {"deterministic": True, "threads": 1},
                "resumed": {"deterministic": False, "threads": 2},
                "note": "x",
            },
            "Note: this segment runs with deterministic False (was True), threads 2 (was 1), "
            "so the remaining steps may not reproduce an uninterrupted run",
        ),
        (
            {"event": "warning", "warning": "cugraph_probe", "message": "cuGraph failed."},
            "Warning: cuGraph failed.",
        ),
        (
            {
                "event": "feature_table",
                "split": "validation",
                "date": "2024-10-01",
                "accounts": 2051,
                "mules": 51,
                "rejected": 0,
            },
            "Feature table, validation at 2024-10-01: 2,051 accounts, 51 mules, 0 rejected",
        ),
        (
            {"event": "diagnose", "analysis": "features", "status": "written", "rows": 6123}
            | {"seconds": 41.2, "finished": "2026-10-03T10:00:00+00:00"},
            "features         written      6,123 rows  41 s",
        ),
        (
            {"event": "diagnose", "analysis": "subgroups", "status": "skipped", "seconds": 0}
            | {"reason": "results/baseline/seed-42 has no audit; run `mule evaluate`"},
            "subgroups        skipped  results/baseline/seed-42 has no audit; run `mule evaluate`",
        ),
        (
            {
                "event": "suite",
                "suite": "controls",
                "dataset_id": "1a2b3c4d5e6f7a8b9c",
                "runs": {
                    "baseline": {"42": "keep", "43": "train"},
                    "no_attention": {"42": "train", "43": "resume"},
                },
                "bound_hours": 12.5,
                "timed_from": "results/baseline/seed-42/history.csv",
            },
            "Suite controls on dataset 1a2b3c4d5e6f: 4 runs, 3 to train, at most 12.5 hours\n"
            "  variant       seed 42   seed 43\n"
            "  baseline      keep      train\n"
            "  no_attention  train     resume",
        ),
        (
            {
                "event": "run_archived",
                "run": str(run.parent.parent / "no_attention" / "seed-43"),
                "archive": str(run.parent.parent / "archive" / "no_attention" / "seed-43" / "T"),
                "differs": ["training.patience"],
            },
            "Moved results/no_attention/seed-43 aside to results/archive/no_attention/seed-43/T "
            "(differs: training.patience)",
        ),
        (
            {"event": "run_finished", "variant": "baseline", "seed": 42, "step": "train"}
            | {"best_epoch": 5, "validation_proxy_ap": 0.5661, "seconds": 8640.0},
            "baseline seed 42 trained: best epoch 5, validation proxy AP 0.566, 2.4 h",
        ),
        (
            {"event": "run_finished", "variant": "baseline", "seed": 42, "step": "audit"}
            | {"validation_ap": 0.3121, "test_ap": None},
            "baseline seed 42 audited: validation AP 0.312, test AP n/a",
        ),
        (
            {"event": "run_failed", "variant": "no_attention", "seed": 43, "step": "train"}
            | {"error": "ValueError: this variant's own failure"},
            "no_attention seed 43 failed in train: ValueError: this variant's own failure",
        ),
        (
            {"event": "suite_stopped", "variant": "no_attention", "seed": 43, "step": "audit"}
            | {"error": "TigerGraphUnavailableError: connect failed after 31 attempt(s)"},
            "TigerGraph stayed unavailable, so the suite stops at no_attention seed 43 (audit): "
            "TigerGraphUnavailableError: connect failed after 31 attempt(s)",
        ),
    ]


def test_each_event_has_its_short_line_or_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for record, text in records(tmp_path / "results" / "baseline" / "seed-42"):
        assert event_text(record) == text, record
    # Running totals, batch counts and scoring progress never reach the console.
    shown = "\n".join(filter(None, (event_text(r) for r, _ in records(tmp_path))))
    for hidden in ("database_calls", "1,234", "roots", "640", "2,011", "752,623", "<html"):
        assert hidden not in shown


def test_events_without_a_line_or_the_fields_their_line_reads_show_nothing() -> None:
    assert event_text({"event": "an_event_no_line_names"}) is None
    # A record from a test or an older version without its line's fields.
    assert event_text({"event": "start", "step": 0}) is None
    assert event_text({"event": "retry", "attempt": 1}) is None


def emitted_events() -> set[str]:
    """Every event name the package emits: the strings of an "event" key's value."""
    names: set[str] = set()
    for path in (REPOSITORY_ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "event":
                    names |= {
                        constant.value
                        for constant in ast.walk(value)
                        if isinstance(constant, ast.Constant) and isinstance(constant.value, str)
                    }
    return names


def test_every_event_the_package_emits_has_a_decided_line() -> None:
    # A new event needs a line, or None, in LINES: what the console shows of it is a choice.
    emitted = emitted_events()
    assert {"start", "resume", "epoch", "retry", "warning", "dataset"} <= emitted
    assert emitted <= set(LINES), emitted - set(LINES)


def test_progress_is_rewritten_in_place_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = Terminal()
    monkeypatch.setattr(sys, "stdout", screen)
    step = {"event": "train", "epoch": 1, "steps": 100, "seconds_per_step": 2.0}
    show_event(step | {"step": 10, "loss": 0.5})
    show_event(step | {"step": 20, "loss": 0.25})
    show("epoch  1  done")
    show_event(step | {"step": 10, "loss": 0.125})
    console.end_progress()
    first = "epoch 1  step 10/100  loss 0.500  2.0 s/step"
    second = "epoch 1  step 20/100  loss 0.250  2.0 s/step"
    assert screen.getvalue() == (
        f"\r{first}\r{second}"
        # A line clears the progress line it replaces.
        f"\r{' ' * len(second)}\repoch  1  done\n"
        "\repoch 1  step 10/100  loss 0.125  2.0 s/step\n"
    )


def test_scoring_is_rewritten_in_place_and_cleared_by_the_next_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = Terminal()
    monkeypatch.setattr(sys, "stdout", screen)
    show_scoring("validation", 640, 2011)
    show_scoring("accounts", 1280)
    show("epoch  1  done")
    first, second = "scoring validation 640/2,011", "scoring accounts 1,280"
    assert screen.getvalue() == (
        f"\r{first}\r{second.ljust(len(first))}\r{' ' * len(second)}\repoch  1  done\n"
    )


def test_a_file_or_pipe_gets_no_progress_lines(capsys: pytest.CaptureFixture[str]) -> None:
    step = {"event": "train", "epoch": 1, "step": 10, "steps": 100}
    show_event(step | {"loss": 0.5, "seconds_per_step": 2.0})
    show_event({"event": "install_wait", "installing": 12, "elapsed_s": 30})
    show_scoring("validation", 640, 2011)
    console.end_progress()
    show("a line")
    assert capsys.readouterr().out == "a line\n"
