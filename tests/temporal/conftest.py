"""Shared setup for the temporal tests.

The fake TigerGraph endpoints live in `temporal_fakes.py` next to this file. Its
directory is put on sys.path here, so test modules can import it under every
pytest import mode (prepend, append and importlib).
"""

from __future__ import annotations

from pathlib import Path
import sys

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
