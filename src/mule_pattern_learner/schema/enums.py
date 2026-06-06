from enum import StrEnum


class VertexType(StrEnum):
    ACCOUNT = "Account"
    PARTY = "Party"
    DEVICE = "Device"
    IP = "IP"
    PHONE = "Phone"
    EMAIL = "Email"
    NAME = "Name"
    BIRTHDATE = "Birthdate"
    STREET_ADDRESS = "Street_Address"


class EdgeType(StrEnum):
    HAS_PAID = "HAS_PAID"
    REV_HAS_PAID = "reverse_HAS_PAID"
    ACCOUNT_ACCOUNT = "Account_Account"
    PARTY_HAS_ACCOUNT = "Party_Has_Account"
    HAS_DEVICE = "Has_Device"
    HAS_IP = "Has_IP"
    HAS_PHONE = "Has_Phone"
    HAS_EMAIL = "Has_Email"
    HAS_NAME = "Has_Name"
    HAS_BIRTHDATE = "Has_Birthdate"
    HAS_STREET_ADDRESS = "Has_Street_Address"


UNDIRECTED_EDGES: frozenset[str] = frozenset(
    {
        EdgeType.ACCOUNT_ACCOUNT.value,
        EdgeType.PARTY_HAS_ACCOUNT.value,
        EdgeType.HAS_DEVICE.value,
        EdgeType.HAS_IP.value,
        EdgeType.HAS_PHONE.value,
        EdgeType.HAS_EMAIL.value,
        EdgeType.HAS_NAME.value,
        EdgeType.HAS_BIRTHDATE.value,
        EdgeType.HAS_STREET_ADDRESS.value,
    }
)

NODE_TYPES: tuple[str, ...] = tuple(vt.value for vt in VertexType)
