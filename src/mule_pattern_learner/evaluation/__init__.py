"""Evaluation, the only code that reads the answer key for a result: the answer-key reader,
mule audit and its checks, test scoring and the fallback list, metrics and the ring
bootstrap, groups, the missed-mule report and identification bias (docs/design.md,
"Evaluation"). No query, data, subgraph, model, training or comparisons code imports it;
only the layers above it (the commands, testing and the benchmark) may, as the import
contracts of pyproject.toml check."""
