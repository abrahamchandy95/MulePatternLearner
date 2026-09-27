"""Fixtures several test modules share.

Shared fakes and builders live in the package, in mule_pattern_learner.testing.
"""

from __future__ import annotations

import pytest

from mule_pattern_learner.tigergraph.render import render_context_query


@pytest.fixture(scope="module")
def text() -> str:
    return render_context_query()
