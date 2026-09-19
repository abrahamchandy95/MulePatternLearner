"""Named experimental protocols and source-availability assumptions."""

from __future__ import annotations
from typing import Any


def validate_protocol(config: dict[str, Any]) -> str:
    protocol = config.get("evaluation_protocol")
    if protocol not in ("strict_inductive", "shared_history"):
        raise ValueError(
            "Specify evaluation_protocol explicitly: strict_inductive or shared_history"
        )
    if protocol == "strict_inductive" and not config.get("scope_id"):
        raise ValueError("Strict inductive sampling requires a frozen TigerGraph scope_id")
    return protocol
