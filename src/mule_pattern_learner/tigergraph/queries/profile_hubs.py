"""profile_hubs: exact whole-world histograms on one Monday of the measures behind read_hubs'
thresholds, for mule bench only (docs/design.md, "profile_hubs")."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

from .rows import as_of, counts, response_head


@dataclass(frozen=True)
class HubProfile:
    """Each histogram maps a measure, distinct customers over the 90 days before t, to the
    number of accounts, devices or IPs that have it: every account that exists at t, by the
    customers owning the other accounts of its payments; and every device and IP with a
    customer, split into registered (a customer has a visible enrolment link to it) and
    observed-only."""

    accounts: dict[int, int]
    registered_devices: dict[int, int]
    observed_devices: dict[int, int]
    registered_ips: dict[int, int]
    observed_ips: dict[int, int]


def params(t_ms: int) -> dict[str, Any]:
    """The parameters of the Monday t."""
    return {"t_ms": as_of(t_ms)}


def _histogram(value: object, what: str, query: str) -> dict[int, int]:
    """A printed histogram: TigerGraph prints a map's integer keys as strings of digits. A
    device or IP is profiled only once it has a customer."""
    found: dict[int, int] = {}
    for key, count in counts(value, what, query).items():
        if not key.isdigit() or count < 1 or (what != "accounts" and int(key) < 1):
            raise ValueError(f"{query} printed in {what} {count} at the measure {key!r}")
        found[int(key)] = count
    return dict(sorted(found.items()))


def parse(rows: Sequence[object]) -> HubProfile:
    """The histograms of the Monday params(t_ms) asked for."""
    from . import PROFILE_HUBS, VERSIONS  # the registry imports this module

    head = response_head(rows, PROFILE_HUBS, VERSIONS[PROFILE_HUBS])
    names = [field.name for field in fields(HubProfile)]
    if set(head) != {"status", "version", *names}:
        raise ValueError(f"{PROFILE_HUBS} printed the fields {sorted(head)}")
    return HubProfile(**{name: _histogram(head[name], name, PROFILE_HUBS) for name in names})
