"""Skip the PyG snapshot-path tests when the optional legacy extra is not installed.

The live temporal trainer needs only torch, so a CUDA host installed with
`.[model,dev,cuda12]` has no torch_geometric; these four modules import it.
"""

import importlib.util

collect_ignore: list[str] = []
if importlib.util.find_spec("torch_geometric") is None:
    collect_ignore = [
        "test_backend.py",
        "test_feature_store.py",
        "test_graph_store.py",
        "test_loop.py",
    ]
