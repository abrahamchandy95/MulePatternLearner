"""The two XGBoost comparisons (docs/design.md, "Comparisons"): XGBoost on the scored account's
own row, and on its row with its customer's, its devices' and IPs' and its community block,
read from the same built batches as the GNN and trained on the same examples, weights and
labels, their round count chosen by training's selection score.

It never imports evaluation or testing, as the import contracts of pyproject.toml check."""
