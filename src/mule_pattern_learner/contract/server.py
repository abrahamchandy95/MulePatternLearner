"""Values the installed TigerGraph queries share with this package.

CONTRACT_VERSION is printed by the context query and recorded by saved models, so it
keeps its value until the query names change on the server.
"""

from __future__ import annotations

CONTRACT_VERSION = "temporal_live_v5_candidate_pools"
