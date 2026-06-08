from dataclasses import dataclass, field

NodeType = str


class NodeIDMapperError(KeyError):
    pass


@dataclass(slots=True)
class NodeIDMapper:
    """Per-batch bidirectional map between global string ids and PyG integer ids.

    This is the TigerGraph analogue of FalkorDB's NodeIDMapper: for one sampled
    batch it assigns each global string id (e.g. "A0000000001") a contiguous
    integer per node type, so the integer can live in PyG's node tensor while
    the FeatureStore can recover the string id to fetch features.

    It is built per batch and bounded by batch size, so no global id table is
    ever materialized regardless of graph scale. Integers are assigned per node
    type starting at 0, mirroring PyG's per-type local indexing.
    """

    _to_int: dict[NodeType, dict[str, int]] = field(default_factory=dict)
    _to_str: dict[NodeType, list[str]] = field(default_factory=dict)

    def register(self, node_type: NodeType, string_ids: list[str]) -> list[int]:
        """Register an ordered list of global string ids for a node type.

        Returns the assigned integer ids, in the same order. Ids already
        registered keep their previously assigned integer; new ids extend the
        mapping. The returned list lines up positionally with string_ids.
        """
        to_int = self._to_int.setdefault(node_type, {})
        to_str = self._to_str.setdefault(node_type, [])
        out: list[int] = []
        for sid in string_ids:
            existing = to_int.get(sid)
            if existing is None:
                new_id = len(to_str)
                to_int[sid] = new_id
                to_str.append(sid)
                out.append(new_id)
            else:
                out.append(existing)
        return out

    def reset(self) -> None:
        """Drop all registered ids, returning the mapper to its empty state.

        The sampler calls this at the start of each batch so the table holds
        only the current batch's nodes, not every node sampled across the run.
        Safe because the loader is synchronous (num_workers=0): the previous
        batch's feature fetch has completed before the next sample begins, so
        no in-flight integer id is invalidated by the reset.
        """
        self._to_int.clear()
        self._to_str.clear()

    def to_string(self, node_type: NodeType, int_id: int) -> str:
        """Recover the global string id for an assigned integer id."""
        ids = self._to_str.get(node_type)
        if ids is None or int_id < 0 or int_id >= len(ids):
            raise NodeIDMapperError(f"no string id for {node_type!r} int id {int_id}")
        return ids[int_id]

    def to_strings(self, node_type: NodeType, int_ids: list[int]) -> list[str]:
        """Recover global string ids for a list of assigned integer ids."""
        return [self.to_string(node_type, i) for i in int_ids]

    def to_int(self, node_type: NodeType, string_id: str) -> int:
        """Look up the integer id assigned to a global string id."""
        mapping = self._to_int.get(node_type)
        if mapping is None or string_id not in mapping:
            raise NodeIDMapperError(f"no int id for {node_type!r} string id {string_id!r}")
        return mapping[string_id]

    def num_nodes(self, node_type: NodeType) -> int:
        """Number of registered ids for a node type."""
        return len(self._to_str.get(node_type, []))

    def node_types(self) -> list[NodeType]:
        """All node types that have registered ids."""
        return list(self._to_str.keys())
