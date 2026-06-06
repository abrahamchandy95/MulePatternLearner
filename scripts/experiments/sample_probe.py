import sys
from typing import cast

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.tigergraph.gsql_paths import gsql_path


def _gsql(client: Client, statement: str) -> str:
    return cast(str, client.conn.gsql(statement))


def main() -> int:
    settings = Settings()
    client = Client(settings)
    print(f"graph: {settings.graphname}\n")

    path = gsql_path("probe_sample_clause")
    if not path.is_file():
        print(f"missing: {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    print("Attempting to CREATE + INSTALL a query that uses 'SAMPLE n EDGE'...\n")
    out = _gsql(client, text + "\nINSTALL QUERY probe_sample_clause\n")

    low = out.lower()
    failed = any(m in low for m in ("error", "fail", "could not", "cannot", "not valid", "syntax"))

    print("=" * 60)
    if failed:
        print("RESULT: SAMPLE clause appears NOT supported (or wrong syntax).")
        print("-> We'll use manual fanout via accumulators.")
        print("=" * 60)
        print("\nServer output (read for the exact reason / suggested syntax):\n")
        print(out)
        return 0

    print("RESULT: SAMPLE clause INSTALLED cleanly -> it IS supported.")
    print("-> We can use SAMPLE for fanout (much simpler).")
    print("=" * 60)
    # run it to confirm it executes and caps edges
    raw = _gsql(client, "USE GRAPH " + settings.graphname + "\nRUN QUERY probe_sample_clause()\n")
    print("\nRun output:\n")
    print(raw[:800])
    return 0


if __name__ == "__main__":
    sys.exit(main())
