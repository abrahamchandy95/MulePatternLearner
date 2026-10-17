"""The Monday calendar, the scored population, the labels reader, the keyed draws, and the
training examples and their weights (docs/design.md, "Worlds, examples and weights").

It reads pu_label only through read_labels, and never imports evaluation or testing, as the
import contracts of pyproject.toml check."""
