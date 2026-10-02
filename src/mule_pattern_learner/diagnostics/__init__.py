"""Analyses of a dataset and its runs against the ground truth; training never imports them.

They read the graph only through ports and truth only through evaluation.truth, and
report tables in the long format of a suite's summary.csv (artifacts.DIAGNOSTIC_TABLES).
`mule diagnose` runs them (diagnostics.study).
"""
