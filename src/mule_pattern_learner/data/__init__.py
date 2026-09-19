"""Data-independent split utilities; label simulation is an external concern."""

from .splitting import Split, SplitConfig, SplitRecord, SplitResult, split_accounts

__all__ = ["Split", "SplitConfig", "SplitRecord", "SplitResult", "split_accounts"]
