"""mule prepare's data work: which configured world is loaded, the weekly calendar and its
check, the scored population, the labels reader, the keyed draws, the training examples and
their weights, the selection and test keys, and the report (docs/design.md, "Worlds,
examples and weights").

It reads pu_label only through read_labels, runs queries only through the fetch executor, and
never imports evaluation or testing, as the import contracts of pyproject.toml check."""
