"""Test support: the reference graph, reference worlds and fixtures, the loader shard
reader, parity, the rules of every query's GSQL text, the planted-slip harness and the
version restamper (docs/design.md, "Package layout"). It may read every column, since it
checks queries and builds no example, batch or model, and nothing that builds them imports
it, as the import contracts of pyproject.toml check."""
