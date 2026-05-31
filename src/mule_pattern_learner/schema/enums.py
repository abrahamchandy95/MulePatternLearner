"""
Graph vocabulary for the Mule_Pattern_Learner graph.

Names match the live TigerGraph schema exactly (case-sensitive). This is the
fraud-relevant subgraph used for account-level mule detection. Transactions are
NOT vertices — payments are the directed, aggregated, time-windowed HAS_PAID
edge (Account -> Account). Account_Account is a separate undirected structural
co-transaction backbone (PageRank / WCC run over it).
"""

from enum import StrEnum


class VertexType(StrEnum):
    ACCOUNT = "Account"
    PARTY = "Party"
    DEVICE = "Device"
    IP = "IP"
    PHONE = "Phone"
    EMAIL = "Email"


class EdgeType(StrEnum):
    # Directed money-flow (aggregated, time-windowed transactions)
    HAS_PAID = "HAS_PAID"
    REV_HAS_PAID = "reverse_HAS_PAID"

    # Undirected structural co-transaction backbone
    ACCOUNT_ACCOUNT = "Account_Account"

    # Undirected identity / ownership
    PARTY_HAS_ACCOUNT = "Party_Has_Account"
    HAS_DEVICE = "Has_Device"
    HAS_IP = "Has_IP"
    HAS_PHONE = "Has_Phone"
    HAS_EMAIL = "Has_Email"


#: TG edge types stored as UNDIRECTED (each expands to a PyG relation pair
#: with a synthetic reverse for symmetric message passing).
UNDIRECTED_EDGES: frozenset[str] = frozenset(
    {
        EdgeType.ACCOUNT_ACCOUNT.value,
        EdgeType.PARTY_HAS_ACCOUNT.value,
        EdgeType.HAS_DEVICE.value,
        EdgeType.HAS_IP.value,
        EdgeType.HAS_PHONE.value,
        EdgeType.HAS_EMAIL.value,
    }
)

NODE_TYPES: tuple[str, ...] = tuple(vt.value for vt in VertexType)
