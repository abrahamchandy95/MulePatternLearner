"""
Per-type feature/label declarations and PyG relation specs for
Mule_Pattern_Learner.

The loader builds from these:

  * pu_label  — the MASKED, realistic label we would actually have in production.
                1 = a CONFIRMED (revealed) mule; 0 = unlabeled (could be a clean
                account OR an undiscovered mule).
  * is_fraud  — the FULL ground truth (all ~216 true mules). EVAL ONLY. Used to
                measure how many HIDDEN mules the model surfaced. NEVER a feature,
                NEVER a training target, NEVER a sampling-bias signal

pu_label has TWO legitimate roles:
  1. Training target  — what the (nnPU) loss is computed against.
  2. Sampling bias     — with so few revealed positives, uniform seed sampling
     yields batches with zero positives and nnPU's positive-risk term never gets
     a gradient. So the loader OVER-SAMPLES revealed-positive seeds (pu_label=1)
     each epoch; standard K-hop neighbor sampling around them pulls in their ring
     co-members (incl. hidden positives), which the model then DISCOVERS.

pu_label is therefore in v_out_labels (the label tensor y) and is read from y by
the sampler/loss. It is NOT added to v_in_feats and NOT duplicated into the extra
attrs.

A model feature may not read fraud labels, directly or transitively.

The model message-passes over the RAW PII-sharing edges (Party -> value
vertex): Has_Device / Has_IP / Has_Phone / Has_Email / Has_Name / Has_Birthdate /
Has_Street_Address. Two parties sharing a value are two hops apart through it —
the "guilt by association" signal. These carry strictly MORE information than the
clusters (sub-threshold links, which value type, how many).
"""

from dataclasses import dataclass
from collections.abc import Mapping

from .enums import UNDIRECTED_EDGES, EdgeType, VertexType

TARGET_VERTEX: str = VertexType.ACCOUNT.value

TARGET_LABEL_ATTR: str = "pu_label"
EVAL_LABEL_ATTR: str = "is_fraud"

SPLIT_ATTRS: tuple[str, ...] = ("is_train", "is_val", "is_test")

AUX_LABEL_ATTRS: tuple[str, ...] = ("fraud", "victim")

ACCOUNT_EXTRA_ATTRS: tuple[str, ...] = SPLIT_ATTRS + AUX_LABEL_ATTRS + (EVAL_LABEL_ATTR,)

ACCOUNT_NUMERIC_FEATURES: tuple[str, ...] = (
    "pagerank",
    "com_size",
    "aa_degree",
    "triangle_count",
    "clustering_coef",
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
    "device_share_cnt",
    "ip_share_cnt",
    "phone_share_cnt",
    "email_share_cnt",
    "is_external",
)

ACCOUNT_EMBEDDING_FEATURE: str = "fastrp_embedding"

LEAKY_EXCLUDED_ATTRS: tuple[str, ...] = (
    "is_fraud",
    "pu_label",
    "fraud",
    "victim",
    "mule_cnt",
    "fraud_ip",
    "fraud_device",
    "trans_in_mule_ratio",
    "trans_out_mule_ratio",
    "shortest_path_length",
    "com_id",
)

DEFERRED_CATEGORICAL_ATTRS: tuple[str, ...] = ("external_type", "category")


VERTEX_FEATURES: Mapping[str, tuple[str, ...]] = {
    VertexType.ACCOUNT.value: ACCOUNT_NUMERIC_FEATURES,
    VertexType.PARTY.value: (),
    VertexType.DEVICE.value: ("is_blocked",),
    VertexType.IP.value: ("is_blocked",),
    VertexType.PHONE.value: (),
    VertexType.EMAIL.value: (),
    VertexType.NAME.value: (),
    VertexType.BIRTHDATE.value: (),
    VertexType.STREET_ADDRESS.value: (),
}


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
_NM = VertexType.NAME
_BD = VertexType.BIRTHDATE
_SA = VertexType.STREET_ADDRESS

_HAS_PAID_ATTRS = (
    "total_amount",
    "total_num_txns",
    "first_txn_date",
    "last_txn_date",
    "span_days",
)

HAS_PAID_SEQUENCE_ATTRS: tuple[str, ...] = ("amount_bins", "count_bins")

HAS_PAID_MAX_BINS: int = 100


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
    _spec(_A, EdgeType.HAS_PAID, _A, _HAS_PAID_ATTRS, time=True),
    _spec(_A, EdgeType.REV_HAS_PAID, _A, _HAS_PAID_ATTRS, time=True),
    _spec(_A, EdgeType.ACCOUNT_ACCOUNT, _A, ("weight",)),
    _spec(_P, EdgeType.PARTY_HAS_ACCOUNT, _A),
    _spec(_P, EdgeType.HAS_DEVICE, _DEV),
    _spec(_P, EdgeType.HAS_IP, _IP),
    _spec(_P, EdgeType.HAS_PHONE, _PH),
    _spec(_P, EdgeType.HAS_EMAIL, _EM),
    _spec(_P, EdgeType.HAS_NAME, _NM),
    _spec(_P, EdgeType.HAS_BIRTHDATE, _BD),
    _spec(_P, EdgeType.HAS_STREET_ADDRESS, _SA),
)


def _expand_undirected(base: tuple[EdgeSpec, ...]) -> tuple[EdgeSpec, ...]:
    """Add synthetic-reverse triples for undirected TG edges (src != dst)."""
    out: list[EdgeSpec] = list(base)
    for s in base:
        if s.name not in UNDIRECTED_EDGES or s.src == s.dst:
            continue
        out.append(EdgeSpec(s.dst, s.name, s.src, s.raw_attrs, s.has_time_attrs, True))
    return tuple(out)


EDGE_SPECS: tuple[EdgeSpec, ...] = _expand_undirected(_BASE_EDGE_SPECS)


def edge_spec(triple: tuple[str, str, str]) -> EdgeSpec:
    for s in EDGE_SPECS:
        if s.triple == triple:
            return s
    raise KeyError(f"Unknown edge triple {triple!r}")


def pytigergraph_v_in_feats() -> dict[str, list[str]]:
    return {vt: list(attrs) for vt, attrs in VERTEX_FEATURES.items()}


def pytigergraph_v_out_labels() -> dict[str, list[str]]:
    """The training target: pu_label (the masked, realistic PU label)."""
    return {TARGET_VERTEX: [TARGET_LABEL_ATTR]}


def pytigergraph_v_extra_feats() -> dict[str, list[str]]:
    """Carried alongside seeds: split masks + aux labels + is_fraud (eval only).
    pu_label is the target (v_out_labels), so it is not repeated here."""
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
