"""
Per-type feature/label declarations and PyG relation specs for
Mule_Pattern_Learner.

Single source of truth for what the model reads from TigerGraph and how it maps
into PyG. The loader builds pyTigerGraph dicts from these; the encoder reads the
edge specs to know which relations carry edge features.

THE FEATURE / LEAKAGE PARTITION (the critical part)
---------------------------------------------------
Target: Account.is_fraud (semantics: "is a mule account"). Treated as
positive-unlabeled — unlabeled accounts (is_fraud=0) are NOT confirmed clean;
they may be undiscovered mules, including whole rings disconnected from any
known fraud.

That constraint imposes a hard rule: a model feature may not read fraud labels,
directly or transitively. The kit's hand-engineered features all violate it
(seeded FROM is_fraud==1, propagated outward) so they (a) leak under an honest
split and (b) are zero for new disconnected rings — useless on the cases we
most need. They are EXCLUDED here and live in TG only for the XGBoost baseline
and eval diagnostics.

Account.fraud / Account.victim (and Party.is_fraud) are likewise never features:
they correlate with the mule label. They are pulled only as masked aux columns
for PU negative-set selection and evaluation context, never as model inputs.
"""

from dataclasses import dataclass
from collections.abc import Mapping

from .enums import UNDIRECTED_EDGES, EdgeType, VertexType

# ── Target ────────────────────────────────────────────────────────────

TARGET_VERTEX: str = VertexType.ACCOUNT.value
TARGET_LABEL_ATTR: str = "is_fraud"  # 1 = mule

#: Split masks (INT 0/1) on the target vertex; written back by the prep step.
SPLIT_ATTRS: tuple[str, ...] = ("is_train", "is_val", "is_test")

#: Masked aux labels carried for PU negative-set selection + eval, NOT features.
AUX_LABEL_ATTRS: tuple[str, ...] = ("fraud", "victim")

#: Everything carried alongside Account seeds beyond input features.
ACCOUNT_EXTRA_ATTRS: tuple[str, ...] = SPLIT_ATTRS + AUX_LABEL_ATTRS


# ── Account feature partition (explicit & auditable) ──────────────────

#: LABEL-FREE numeric Account features used as model inputs. Generalize to new
#: disconnected rings because none reads is_fraud.
ACCOUNT_NUMERIC_FEATURES: tuple[str, ...] = (
    # structural (Account_Account: PageRank / WCC / degree / motifs)
    "pagerank",
    "com_size",
    "aa_degree",
    "triangle_count",
    "clustering_coef",
    # money-flow (HAS_PAID directed aggregates)
    "in_degree",
    "out_degree",
    "in_amount",
    "out_amount",
    "in_txn_count",
    "out_txn_count",
    "fan_in_ratio",
    "fan_out_ratio",
    "pass_through_ratio",
    "net_flow",
    # temporal
    "activity_span_days",
    "days_since_last_txn",
    "account_age_days",
    "mean_inter_txn_days",
    "txn_per_active_day",
    "burst_ratio",
    "active_bin_count",
    "activity_concentration",
    "peak_bin_fraction",
    "early_late_ratio",
    "in_out_lag_days",
    # identity-sharing density (label-free: counts ALL sharing accounts)
    "device_share_cnt",
    "ip_share_cnt",
    "phone_share_cnt",
    "email_share_cnt",
    # external-counterparty flag (label-free, structural)
    "is_external",
)

#: Optional unsupervised structural embedding (reserved; LIST<DOUBLE>). Pulled
#: as a vector feature only when the ablation enables it.
ACCOUNT_EMBEDDING_FEATURE: str = "fastrp_embedding"

#: Account attributes EXCLUDED from model inputs — leaky (read is_fraud
#: directly/transitively) or ring identifiers (memorization). Documented so the
#: choice is reviewable and never silently reversed.
LEAKY_EXCLUDED_ATTRS: tuple[str, ...] = (
    "is_fraud",  # target
    "fraud",  # aux label
    "victim",  # aux label
    "mule_cnt",  # n-hop count of known mules
    "fraud_ip",  # n-hop fraud reachable via shared IP
    "fraud_device",  # n-hop fraud reachable via shared device
    "trans_in_mule_ratio",
    "trans_out_mule_ratio",
    "shortest_path_length",  # distance to nearest known fraud
    "com_id",  # ring identifier
)

#: Categorical Account attrs deferred (need encoding before use): external_type,
#: category. Not in the numeric feature set yet.
DEFERRED_CATEGORICAL_ATTRS: tuple[str, ...] = ("external_type", "category")


# ── Vertex input features per type ────────────────────────────────────
#
# Only numeric, label-free attributes. Empty tuple = featureless (the encoder
# substitutes a learnable per-type embedding). Device/IP carry the label-free
# is_blocked risk flag. Party/Phone/Email are pure relays.

VERTEX_FEATURES: Mapping[str, tuple[str, ...]] = {
    VertexType.ACCOUNT.value: ACCOUNT_NUMERIC_FEATURES,
    VertexType.PARTY.value: (),
    VertexType.DEVICE.value: ("is_blocked",),
    VertexType.IP.value: ("is_blocked",),
    VertexType.PHONE.value: (),
    VertexType.EMAIL.value: (),
}


# ── Edge spec ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """One directed PyG relation.

    src/name/dst form the PyG triple; name matches the TG edge id.
    raw_attrs are the TG edge attributes pulled into edge_attr (pre-transform).
    has_time_attrs flags trailing DATETIME attrs needing epoch->feature
    conversion. synthetic_reverse marks a reverse direction we add for an
    UNDIRECTED TG edge.
    """

    src: str
    name: str
    dst: str
    raw_attrs: tuple[str, ...] = ()
    has_time_attrs: bool = False
    synthetic_reverse: bool = False

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.src, self.name, self.dst)


_A = VertexType.ACCOUNT
_P = VertexType.PARTY
_DEV = VertexType.DEVICE
_IP = VertexType.IP
_PH = VertexType.PHONE
_EM = VertexType.EMAIL

#: HAS_PAID scalar edge attributes, in TG column order. The two trailing dates
#: are converted to recency/duration features by models.edge_transform. The
#: bi-weekly bin sequences (amount_bins, count_bins) are LIST<...> attributes
#: handled separately by the loader (padded/truncated to a fixed bin count for
#: PyG), not part of the fixed scalar edge_attr vector.
_HAS_PAID_ATTRS = (
    "total_amount",
    "total_num_txns",
    "first_txn_date",
    "last_txn_date",
    "span_days",
)

#: HAS_PAID temporal sequence attributes (LIST). Loaded and padded/truncated to
#: HAS_PAID_MAX_BINS by the loader; consumed flat or by a sequence encoder.
HAS_PAID_SEQUENCE_ATTRS: tuple[str, ...] = ("amount_bins", "count_bins")

#: Fixed bin count the loader pads/truncates the sequences to when forming a
#: dense per-edge tensor for PyG (covers up to ~64 weeks at bi-weekly width).
HAS_PAID_MAX_BINS: int = 32


def _spec(
    src: VertexType,
    name: EdgeType,
    dst: VertexType,
    attrs: tuple[str, ...] = (),
    time: bool = False,
    syn: bool = False,
) -> EdgeSpec:
    return EdgeSpec(src.value, name.value, dst.value, attrs, time, syn)


_BASE_EDGE_SPECS: tuple[EdgeSpec, ...] = (
    # Directed money-flow (real TG edges; both carry the aggregates)
    _spec(_A, EdgeType.HAS_PAID, _A, _HAS_PAID_ATTRS, time=True),
    _spec(_A, EdgeType.REV_HAS_PAID, _A, _HAS_PAID_ATTRS, time=True),
    # Undirected structural backbone (forward; reverse expanded below)
    _spec(_A, EdgeType.ACCOUNT_ACCOUNT, _A, ("weight",)),
    # Undirected identity / ownership (forward)
    _spec(_P, EdgeType.PARTY_HAS_ACCOUNT, _A),
    _spec(_P, EdgeType.HAS_DEVICE, _DEV),
    _spec(_P, EdgeType.HAS_IP, _IP),
    _spec(_P, EdgeType.HAS_PHONE, _PH),
    _spec(_P, EdgeType.HAS_EMAIL, _EM),
)


def _expand_undirected(base: tuple[EdgeSpec, ...]) -> tuple[EdgeSpec, ...]:
    """Add synthetic-reverse triples for undirected TG edges (src != dst)."""
    out: list[EdgeSpec] = list(base)
    for s in base:
        if s.name not in UNDIRECTED_EDGES or s.src == s.dst:
            continue  # directed edges + self-relations need no synthetic reverse
        out.append(EdgeSpec(s.dst, s.name, s.src, s.raw_attrs, s.has_time_attrs, True))
    return tuple(out)


EDGE_SPECS: tuple[EdgeSpec, ...] = _expand_undirected(_BASE_EDGE_SPECS)


# ── Lookup / loader-dict builders ─────────────────────────────────────


def edge_spec(triple: tuple[str, str, str]) -> EdgeSpec:
    for s in EDGE_SPECS:
        if s.triple == triple:
            return s
    raise KeyError(f"Unknown edge triple {triple!r}")


def pytigergraph_v_in_feats() -> dict[str, list[str]]:
    return {vt: list(attrs) for vt, attrs in VERTEX_FEATURES.items()}


def pytigergraph_v_out_labels() -> dict[str, list[str]]:
    return {TARGET_VERTEX: [TARGET_LABEL_ATTR]}


def pytigergraph_v_extra_feats() -> dict[str, list[str]]:
    return {TARGET_VERTEX: list(ACCOUNT_EXTRA_ATTRS)}


def pytigergraph_e_in_feats() -> dict[str, list[str]]:
    """Scalar edge attributes per TG edge-type name (the fixed-width features
    that become PyG edge_attr directly). Reverses reuse the same name, so
    de-dup by name. Does NOT include the HAS_PAID bin sequences — those are
    variable-length LISTs fetched separately (see e_sequence_feats)."""
    out: dict[str, list[str]] = {}
    for s in EDGE_SPECS:
        if s.raw_attrs and s.name not in out:
            out[s.name] = list(s.raw_attrs)
    return out


def pytigergraph_e_sequence_feats() -> dict[str, list[str]]:
    """Variable-length LIST edge attributes per TG edge-type name. The loader
    pulls these and pads/truncates each to HAS_PAID_MAX_BINS to form a dense
    per-edge sequence tensor (consumed flat or by a sequence encoder). Kept
    separate from e_in_feats because they need different handling than scalar
    edge_attr."""
    return {EdgeType.HAS_PAID.value: list(HAS_PAID_SEQUENCE_ATTRS)}


def pyg_metadata() -> tuple[list[str], list[tuple[str, str, str]]]:
    return list(VERTEX_FEATURES.keys()), [s.triple for s in EDGE_SPECS]
