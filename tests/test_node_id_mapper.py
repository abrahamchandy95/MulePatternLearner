from __future__ import annotations

import pytest

from mule_pattern_learner.pyg.node_id_mapper import (
    NodeIDMapper,
    NodeIDMapperError,
)


class TestRegister:
    def test_assigns_contiguous_ints_from_zero(self) -> None:
        mapper = NodeIDMapper()
        ints = mapper.register("Account", ["A1", "A2", "A3"])
        assert ints == [0, 1, 2]

    def test_per_type_independent_numbering(self) -> None:
        mapper = NodeIDMapper()
        acct = mapper.register("Account", ["A1", "A2"])
        party = mapper.register("Party", ["C1", "C2"])
        # both types start their own numbering at 0
        assert acct == [0, 1]
        assert party == [0, 1]

    def test_duplicate_ids_keep_assignment(self) -> None:
        mapper = NodeIDMapper()
        first = mapper.register("Account", ["A1", "A2"])
        again = mapper.register("Account", ["A2", "A3"])
        assert first == [0, 1]
        # A2 keeps id 1, A3 is new -> 2
        assert again == [1, 2]
        assert mapper.num_nodes("Account") == 3


class TestRoundTrip:
    def test_int_to_string_and_back(self) -> None:
        mapper = NodeIDMapper()
        _ = mapper.register("Account", ["A1", "A2", "A3"])
        assert mapper.to_string("Account", 0) == "A1"
        assert mapper.to_string("Account", 2) == "A3"
        assert mapper.to_int("Account", "A2") == 1

    def test_to_strings_list(self) -> None:
        mapper = NodeIDMapper()
        _ = mapper.register("Account", ["A1", "A2", "A3"])
        assert mapper.to_strings("Account", [2, 0]) == ["A3", "A1"]

    def test_entity_integer_string_ids(self) -> None:
        # Resolved_Entity ids are integer-strings; mapper treats them as strings.
        mapper = NodeIDMapper()
        ints = mapper.register("Resolved_Entity", ["59770824", "42993017"])
        assert ints == [0, 1]
        assert mapper.to_string("Resolved_Entity", 1) == "42993017"


class TestErrors:
    def test_unknown_int_raises(self) -> None:
        mapper = NodeIDMapper()
        _ = mapper.register("Account", ["A1"])
        with pytest.raises(NodeIDMapperError):
            _ = mapper.to_string("Account", 5)

    def test_unknown_string_raises(self) -> None:
        mapper = NodeIDMapper()
        _ = mapper.register("Account", ["A1"])
        with pytest.raises(NodeIDMapperError):
            _ = mapper.to_int("Account", "A_MISSING")

    def test_unknown_type_raises(self) -> None:
        mapper = NodeIDMapper()
        with pytest.raises(NodeIDMapperError):
            _ = mapper.to_string("Party", 0)


class TestIntrospection:
    def test_num_nodes_and_types(self) -> None:
        mapper = NodeIDMapper()
        _ = mapper.register("Account", ["A1", "A2"])
        _ = mapper.register("Party", ["C1"])
        assert mapper.num_nodes("Account") == 2
        assert mapper.num_nodes("Party") == 1
        assert mapper.num_nodes("Missing") == 0
        assert set(mapper.node_types()) == {"Account", "Party"}
