from __future__ import annotations

from dataclasses import dataclass


class NeighborFanoutError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NeighborFanout:
    """Per-relation neighbor counts for the two-round k-hop sampler.

    A parameter object (in PyG's NumNeighbors spirit): a small, typed, validated
    bundle of how many neighbors are drawn per relation per round. It is the
    single source of truth for sampler fanout. Round-2 counts are intentionally
    smaller than round-1 to control neighborhood explosion (standard
    decreasing-fanout practice).

    Field names describe the relation; the mapping to the GSQL query's parameter
    names (fanout_has_paid, ...) lives only in as_query_params().
    """

    has_paid: int = 15
    account_account: int = 10
    acct_party: int = 5
    party_entity: int = 5
    entity_party: int = 10
    party_acct: int = 10
    has_paid_round2: int = 5
    account_account_round2: int = 5
    acct_party_round2: int = 3
    party_entity_round2: int = 3
    entity_party_round2: int = 5
    party_acct_round2: int = 5

    def __post_init__(self) -> None:
        for name, value in self._counts().items():
            if isinstance(value, bool):
                raise NeighborFanoutError(f"{name}: fanout must be an int, got bool")
            if value < 1:
                raise NeighborFanoutError(f"{name}: fanout must be >= 1, got {value}")

    def _counts(self) -> dict[str, int]:
        """All fanout fields as an explicitly typed name -> value mapping."""
        return {
            "has_paid": self.has_paid,
            "account_account": self.account_account,
            "acct_party": self.acct_party,
            "party_entity": self.party_entity,
            "entity_party": self.entity_party,
            "party_acct": self.party_acct,
            "has_paid_round2": self.has_paid_round2,
            "account_account_round2": self.account_account_round2,
            "acct_party_round2": self.acct_party_round2,
            "party_entity_round2": self.party_entity_round2,
            "entity_party_round2": self.entity_party_round2,
            "party_acct_round2": self.party_acct_round2,
        }

    def as_query_params(self) -> dict[str, int]:
        """Return the fanouts keyed by the GSQL query's parameter names.

        The sample_khop_neighborhood query names its round-2 parameters with a
        trailing '_2'; that GSQL-specific spelling is confined to this method.
        """
        return {
            "fanout_has_paid": self.has_paid,
            "fanout_account_account": self.account_account,
            "fanout_acct_party": self.acct_party,
            "fanout_party_entity": self.party_entity,
            "fanout_entity_party": self.entity_party,
            "fanout_party_acct": self.party_acct,
            "fanout_has_paid_2": self.has_paid_round2,
            "fanout_account_account_2": self.account_account_round2,
            "fanout_acct_party_2": self.acct_party_round2,
            "fanout_party_entity_2": self.party_entity_round2,
            "fanout_entity_party_2": self.entity_party_round2,
            "fanout_party_acct_2": self.party_acct_round2,
        }
