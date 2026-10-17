# Mule Pattern Learner

Finds money-mule accounts that a bank has not found yet with a temporal heterogeneous graph
neural network on TigerGraph, trained and scored on data simulated by PhantomLedger.

**The code is being rebuilt.** What it will do, and every decision behind it, is in
[the design](docs/design.md). Until the rebuild is done the package holds only what the
design keeps from the earlier code: the TigerGraph client and its retrying query executor,
the nnPU loss, and the weighted metrics with the ring bootstrap. The
`mule` command names the design's commands, and each one still says it is not built yet.
The earlier code is in this repository's history.

## Setup

Python 3.12 or newer in a `.venv`, installed editable:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set `HOST`, `GRAPHNAME` and `SECRET`; environment
variables override it.

## Development

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/basedpyright
.venv/bin/pytest -q
```

The tests never connect to TigerGraph: tests that need it are marked `graph` and run only
with `pytest -m graph`. The import contracts in `pyproject.toml` run as one of the tests.

## License

MIT (author: Abraham Chandy).
