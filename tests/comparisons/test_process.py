"""XGBoost in a Python process of its own (comparisons.boosting and comparisons.process,
docs/design.md, "Comparisons").

torch 2.12 and XGBoost 3.2.0 load different OpenMP runtimes, and an XGBoost fit with more than
one thread crashes a process that has torch loaded. So the module that imports xgboost imports
nothing that imports torch, and every fit and prediction runs in a child that never imports
torch, at XGBoost's full thread count. XGBoost is never imported in the test process, which has
torch loaded: whatever must call it in this process runs in a fresh interpreter without torch,
and no test crashes Python on purpose."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

import numpy as np
import pytest
import torch

from mule_pattern_learner.comparisons import process
from mule_pattern_learner.comparisons.process import XGBoostProcessError, exit_message


def fresh(*parts: str) -> str:
    """The output of a script of parts, each dedented, run by a fresh interpreter, which must
    succeed."""
    script = "\n".join(textwrap.dedent(part) for part in parts)
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _planted(n: int = 300) -> tuple[Any, Any, Any, list[str]]:
    """n rows of 6 columns, positive when column c2 is above 1 and noise does not hide it."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 6)).astype(np.float32)
    labels = (x[:, 2] + rng.normal(size=n) * 0.4 > 1.0).astype(np.int64)
    return x, labels, 1.0 + rng.random(n), [f"c{index}" for index in range(6)]


PLANTED = """
import sys
import numpy as np
import pandas as pd
import xgboost
from mule_pattern_learner.comparisons import boosting, process
from mule_pattern_learner.comparisons.rounds import MAX_ROUNDS, equal_totals

rng = np.random.default_rng(7)
n = 300
x = rng.normal(size=(n, 6)).astype(np.float32)
labels = (x[:, 2] + rng.normal(size=n) * 0.4 > 1.0).astype(np.int64)
weights = 1.0 + rng.random(n)
columns = [f"c{index}" for index in range(6)]
"""


def test_importing_the_xgboost_module_leaves_torch_out_of_a_fresh_interpreter() -> None:
    found = fresh(
        """
        import sys
        import mule_pattern_learner.comparisons.boosting
        print(sorted(name for name in ("torch", "torch_geometric") if name in sys.modules))
        print("xgboost" in sys.modules)
        """
    )
    assert found.splitlines() == ["[]", "True"]


def test_nothing_that_holds_torch_imports_xgboost() -> None:
    """The comparisons' ranker and vectors, the clock probe and the evaluation report hold
    torch, so none of them, nor anything they import, imports xgboost."""
    found = fresh(
        """
        import sys
        import mule_pattern_learner.cli
        import mule_pattern_learner.comparisons.features
        import mule_pattern_learner.comparisons.process
        import mule_pattern_learner.comparisons.rankers
        import mule_pattern_learner.evaluation.audit
        import mule_pattern_learner.evaluation.clock_probe
        import mule_pattern_learner.evaluation.report
        print("torch" in sys.modules, "xgboost" in sys.modules)
        """
    )
    assert found == "True False"


def test_the_process_module_imports_neither_torch_nor_xgboost() -> None:
    found = fresh(
        """
        import sys
        import mule_pattern_learner.comparisons.process
        print("torch" in sys.modules, "xgboost" in sys.modules)
        """
    )
    assert found == "False False"


def test_a_fit_and_a_prediction_run_in_a_child_beside_a_busy_torch() -> None:
    """torch has run an operation in this process, whose XGBoost it would crash, and a fit and
    a prediction through the child process learn the planted cue all the same."""
    busy = torch.randn(512, 512) @ torch.randn(512, 512)
    assert busy.shape == (512, 512)
    x, labels, weights, columns = _planted()
    booster = process.fit(x, labels, weights, columns)
    found = process.scores(booster, x, columns, 20)
    assert found.shape == (len(x),) and found.dtype == np.float64 and np.isfinite(found).all()
    assert found[labels == 1].mean() > found[labels == 0].mean() + 0.5
    assert "xgboost" not in sys.modules


def test_the_childs_booster_and_scores_equal_an_in_process_fit() -> None:
    """On the same data, the child's booster is byte for byte the one XGBoost fits straight in
    a process without torch, and so are its scores at several round counts."""
    found = fresh(
        PLANTED,
        """
        # In this process, straight through XGBoost, with the weights at equal totals.
        model = xgboost.XGBClassifier(n_estimators=MAX_ROUNDS)
        model.fit(
            pd.DataFrame(x, columns=columns), labels, sample_weight=equal_totals(labels, weights)
        )
        direct = bytes(model.get_booster().save_raw("ubj"))
        # Through the child process.
        child = process.fit(x, labels, weights, columns)
        assert child == direct, "the boosters differ"
        for rounds in (10, 370, 1000):
            here = boosting.scores(direct, x, columns, rounds)
            there = process.scores(child, x, columns, rounds)
            assert np.array_equal(here, there), rounds
        assert "torch" not in sys.modules
        print("equal")
        """,
    )
    assert found == "equal"


def test_at_xgboosts_defaults_neither_a_seed_nor_a_thread_count_changes_the_booster() -> None:
    """The premise of training each comparison once with no seed (docs/design.md, "Comparisons"):
    at XGBoost's default settings every row and column is used by every tree, so no seed has
    anything to draw and the booster is byte for byte the same with no seed, seed 0 and seed 1,
    and on one thread or four. A default that drew a subsample would make the one model of a
    comparison an arbitrary draw, and this test would fail on the library that changed it."""
    found = fresh(
        PLANTED,
        """
        frame = pd.DataFrame(x, columns=columns)
        weighted = equal_totals(labels, weights)
        boosters = set()
        for seed in (None, 0, 1):
            for jobs in (1, 4):
                model = xgboost.XGBClassifier(n_estimators=50, random_state=seed, n_jobs=jobs)
                model.fit(frame, labels, sample_weight=weighted)
                boosters.add(bytes(model.get_booster().save_raw("ubj")))
        assert len(boosters) == 1, f"{len(boosters)} boosters from 6 seeds and thread counts"
        print("identical")
        """,
    )
    assert found == "identical"


def test_the_clock_probes_folds_equal_an_in_process_fit() -> None:
    found = fresh(
        PLANTED,
        """
        positive = labels.astype(bool)
        fold = (np.arange(n) % 2).astype(np.int64)
        direct = np.zeros(n)
        for held in range(2):
            train = fold != held
            model = xgboost.XGBClassifier(random_state=3)
            model.fit(x[train], positive[train].astype(np.int64))
            direct[~train] = model.predict(x[~train], output_margin=True)
        child = process.fold_scores(x.astype(np.float64), positive, fold, 3)
        assert np.array_equal(child, direct)
        assert np.array_equal(boosting.fold_scores(x.astype(np.float64), positive, fold, 3), child)
        print("equal")
        """,
    )
    assert found == "equal"


def test_the_choice_of_a_round_count_in_the_child_equals_one_in_process() -> None:
    found = fresh(
        PLANTED,
        """
        keys = pd.DataFrame(
            {
                "t_ms": np.zeros(n, dtype=np.int64),
                "account_id": [f"a{index:04d}" for index in range(n)],
                "positive": labels.astype(bool),
                "ranked": True,
                "weight": 1.0,
            }
        )
        raw = process.fit(x, labels, weights, columns)
        here = boosting.choose(raw, x, columns, keys)
        there = process.choose(raw, x, columns, keys)
        assert here[0] == there[0] and here[2] == there[2] and len(there[2]) == 100
        assert here[1].accounts == there[1].accounts
        assert np.array_equal(here[1].weights, there[1].weights)
        assert np.array_equal(here[1].shares, there[1].shares)
        print("equal")
        """,
    )
    assert found == "equal"


def test_the_fit_keeps_xgboosts_defaults_and_names_its_columns() -> None:
    """XGBoost's default settings with 1,000 rounds and no seed, one tree per round."""
    found = fresh(
        PLANTED,
        """
        made = []
        real = xgboost.XGBClassifier

        def recording(**keywords):
            made.append(keywords)
            return real(**keywords)

        boosting.xgboost.XGBClassifier = recording
        raw = boosting.fit(x, labels, equal_totals(labels, weights), columns)
        assert made == [{"n_estimators": MAX_ROUNDS}], made
        booster = xgboost.Booster()
        booster.load_model(bytearray(raw))
        assert booster.num_boosted_rounds() == MAX_ROUNDS and booster.feature_names == columns
        print("defaults")
        """,
    )
    assert found == "defaults"


def test_a_score_is_the_sum_of_the_trees_up_to_its_round() -> None:
    """XGBoost's margin up to a round, less its base score, which every row shares."""
    found = fresh(
        PLANTED,
        """
        raw = boosting.fit(x, labels, equal_totals(labels, weights), columns)
        booster = xgboost.Booster()
        booster.load_model(bytearray(raw))
        frame = pd.DataFrame(x, columns=columns)
        for rounds in (10, 30):
            margin = booster.predict(
                xgboost.DMatrix(frame), output_margin=True, iteration_range=(0, rounds)
            )
            found = boosting.scores(raw, x, columns, rounds)
            assert found.dtype == np.float64
            offset = margin.astype(np.float64) - found
            np.testing.assert_allclose(offset, offset[0], atol=1e-5)
        print("sums")
        """,
    )
    assert found == "sums"


def test_the_parent_refuses_what_it_can_before_starting_a_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never(*args: Any, **keywords: Any) -> None:
        raise AssertionError("a child process was started")

    monkeypatch.setattr(process.subprocess, "run", never)
    x, labels, weights, columns = _planted(20)
    with pytest.raises(ValueError, match="6 wide, not 5"):
        process.fit(x, labels, weights, columns[:5])
    with pytest.raises(ValueError, match="no label 1"):
        process.fit(x, np.zeros(len(x), dtype=np.int64), weights, columns)
    with pytest.raises(ValueError, match="multiple of 10"):
        process.scores(b"", x, columns, 25)


def test_a_child_that_fails_raises_a_clear_error_in_the_parent() -> None:
    x, _, _, columns = _planted(20)
    with pytest.raises(XGBoostProcessError) as caught:
        process.scores(b"this is not a booster", x, columns, 10)
    message = str(caught.value)
    assert message.startswith("The XGBoost process for scores exited with status 1:")
    assert "XGBoostError" in message
    # Nothing it holds is lost to the caller, and it is a RuntimeError.
    assert isinstance(caught.value, RuntimeError)
    with pytest.raises(XGBoostProcessError, match="exited with status 2:\nusage: python -m"):
        process.run("no_such_operation", {})


def test_how_a_child_ended_is_named_without_crashing_anything() -> None:
    errors = (
        "Traceback (most recent call last):\n"
        '  File "boosting.py", line 1, in main\n'
        "    raise ValueError('boom')\n"
        "ValueError: boom\n"
        "second line of it\n"
    )
    found = exit_message("fit", 1, errors)
    assert (
        found
        == "The XGBoost process for fit exited with status 1:\nValueError: boom\nsecond line of it"
    )
    assert exit_message("fit", -11, "").startswith(
        "The XGBoost process for fit was killed by SIGSEGV:"
    )
    assert "it printed no error" in exit_message("fit", -9, "")
    assert "killed by signal 999" in exit_message("fit", -999, "")
    # With no traceback, the last lines show.
    assert process.error_lines("one\n\ntwo\n") == ["one", "two"]
    # Of two tracebacks, the last one's exception.
    chained = (
        errors
        + "\nDuring handling:\n\nTraceback (most recent call last):\n  File x\nKeyError: 'k'\n"
    )
    assert process.error_lines(chained) == ["KeyError: 'k'"]


def test_a_jobs_folder_is_removed_whether_the_child_succeeds_or_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    x, labels, weights, columns = _planted(60)
    booster = process.fit(x, labels, weights, columns)
    process.scores(booster, x, columns, 10)
    with pytest.raises(XGBoostProcessError):
        process.scores(b"not a booster", x, columns, 10)
    assert list(tmp_path.iterdir()) == []


def test_the_child_is_started_as_the_boosting_module_with_the_parents_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XGBoost keeps its full thread count: the child is given no limit on OpenMP, and the
    environment it inherits is the parent's own."""
    calls: list[tuple[list[str], dict[str, Any]]] = []
    real = subprocess.run

    def recording(arguments: list[str], **keywords: Any) -> Any:
        calls.append((arguments, keywords))
        return real(arguments, **keywords)

    monkeypatch.setattr(process.subprocess, "run", recording)
    x, labels, weights, columns = _planted(40)
    process.fit(x, labels, weights, columns)
    ((arguments, keywords),) = calls
    assert arguments[:3] == [sys.executable, "-m", process.MODULE]
    assert arguments[3] == "fit"
    assert "env" not in keywords
    assert process.MODULE == "mule_pattern_learner.comparisons.boosting"
