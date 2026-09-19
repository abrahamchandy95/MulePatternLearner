"""Regenerate the temporal context query after editing its shared contract."""

from pathlib import Path

from mule_pattern_learner.temporal.live.queries import render_context_query


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    (root / "gsql/temporal/training_context.gsql").write_text(render_context_query())


if __name__ == "__main__":
    main()
