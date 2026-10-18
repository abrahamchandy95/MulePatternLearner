"""The encoding of a batch: every column kind as docs/design.md's "Encoding" gives it, each
column in its place, the widths of docs/features.md's "Widths", the batch's other tensors, the
device it is encoded on, and the Fourier basis at the durations the design names."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import Tensor

from mule_pattern_learner.model.encode import Kind, Layout, encode, encode_batch
from mule_pattern_learner.model.time_encoding import SCALE, WIDTH, encode_reference
from mule_pattern_learner.subgraph.build import Batch
from mule_pattern_learner.subgraph.layouts import (
    ACCOUNT_LAYOUT,
    COMMUNITY_LAYOUT,
    LAYOUT,
    OWNED_BY,
    PAYMENT_LAYOUT,
    PAYS,
    SEEN_LAYOUT,
    reverse,
)

SECOND_MS, DAY_MS = 1_000, 86_400_000
DEVICES = [
    "cpu",
    *(["cuda"] if torch.cuda.is_available() else []),
    *(["mps"] if torch.backends.mps.is_available() else []),
]


def _fourier(duration_ms: int) -> list[float]:
    return encode_reference(np.array(duration_ms)).tolist()


def _encoded(layout: Layout, raw: list[float]) -> Tensor:
    return encode(torch.tensor([raw], dtype=torch.float64), layout)[0]


def test_a_payment_edge_is_encoded_column_by_column() -> None:
    """USD 19.99 by card, 2 days old, after 2 earlier pair payments, the latest 30 days
    before it; no pair age, sender gap or receiver gap given; tracked, 25% left within a
    day, the dwell absent."""
    card = [0, 1, 0, 0, 0, 0, 0]
    raw = [1_999, *card, 1, 2 * DAY_MS, 2, 1, 30 * DAY_MS, 0, 0, 0, 0, 0, 0, 1, 0.25, 0, 0]
    found = _encoded(PAYMENT_LAYOUT, raw)
    assert found.shape == (401,) and found.dtype == torch.float32
    absent = [0.0] * (1 + WIDTH)
    expected = [
        math.log1p(19.99),
        *card,
        1,
        *_fourier(2 * DAY_MS),
        math.log1p(2),
        1,
        *_fourier(30 * DAY_MS),
        *absent,
        *absent,
        *absent,
        1,
        0.25,
        *absent,
    ]
    np.testing.assert_allclose(found.numpy(), expected, atol=1e-5)


def test_a_seen_edge_and_a_community_block() -> None:
    found = _encoded(SEEN_LAYOUT, [4, 1, 40 * DAY_MS, 1, 4 * DAY_MS])
    expected = [math.log1p(4), 1, *_fourier(40 * DAY_MS), 1, *_fourier(4 * DAY_MS)]
    np.testing.assert_allclose(found.numpy(), expected, atol=1e-5)
    # Members and payments are counts, internal cents an amount, the rest as they are.
    block = _encoded(COMMUNITY_LAYOUT, [3, 1, 1, 0.75, 0.5, 47_500, 0.25, 4])
    np.testing.assert_allclose(
        block.numpy(),
        [math.log1p(3), 1, math.log1p(1), 0.75, 0.5, math.log1p(475), 0.25, math.log1p(4)],
        rtol=1e-6,
    )


def test_every_account_column_keeps_its_place(raw_batch: Batch) -> None:
    """Each column of the account layout, read by name from a raw row, is at its encoded
    place: an amount as log(1 + dollars), a count as log(1 + n), a share or flag as it is, a
    duration as its bit and 64 Fourier numbers."""
    raw = raw_batch.data["account"].x
    encoded = encode(raw, ACCOUNT_LAYOUT)
    assert encoded.shape == (raw.shape[0], 309)
    out = 0
    for column in ACCOUNT_LAYOUT.columns:
        values = raw[:, ACCOUNT_LAYOUT.index(column.raw_names[0])]
        if column.kind is Kind.DURATION:
            present = values
            ms = raw[:, ACCOUNT_LAYOUT.index(column.raw_names[1])].to(torch.int64)
            fourier = torch.as_tensor(encode_reference(ms.numpy()), dtype=torch.float32)
            expected = torch.cat([present[:, None].float(), fourier * present[:, None].float()], 1)
        elif column.kind is Kind.AMOUNT:
            expected = torch.log1p(values / 100).float()[:, None]
        elif column.kind is Kind.COUNT:
            expected = torch.log1p(values).float()[:, None]
        else:
            expected = values.float()[:, None]
        torch.testing.assert_close(
            encoded[:, out : out + column.width], expected, atol=1e-5, rtol=0
        )
        out += column.width
    assert out == 309


def test_a_batch_is_encoded_to_the_widths_of_the_features(raw_batch: Batch) -> None:
    data = encode_batch(raw_batch.data, LAYOUT)
    widths = {kind: data[kind].x.shape[1] for kind in data.node_types}
    assert widths == {"account": 309, "customer": 11, "device": 12, "ip": 12}
    edge_widths = {
        relation[1]: data[relation].edge_attr.shape[1]
        for relation in data.edge_types
        if "edge_attr" in data[relation]
    }
    expected: dict[str, int] = {}
    for _, name, _ in LAYOUT.relations:
        base = name.removeprefix("rev_")
        if base != "owned_by":
            expected[name] = 401 if base == "pays" else 131 if base.startswith("seen_on") else 130
    assert edge_widths == expected
    assert len(edge_widths) == 14
    assert "edge_attr" not in data[OWNED_BY] and "edge_attr" not in data[reverse(OWNED_BY)]
    # A reverse relation reads its forward relation's encoded attributes.
    assert data[reverse(PAYS)].edge_attr is data[PAYS].edge_attr
    assert data.community.shape == (32, 8) and data.community.dtype == torch.float32
    assert torch.equal(data.scored_index, raw_batch.data.scored_index)
    assert torch.equal(data.label, raw_batch.data.label)
    assert data.weight.dtype == torch.float32 and data.weight.tolist() == [1.0] * 32
    for relation in data.edge_types:
        assert torch.equal(data[relation].edge_index, raw_batch.data[relation].edge_index)


def test_a_scoring_batch_is_encoded_without_labels_and_weights(raw_batch: Batch) -> None:
    scoring = raw_batch.data.clone()
    del scoring.label, scoring.weight
    data = encode_batch(scoring, LAYOUT)
    assert "label" not in data and "weight" not in data
    training = encode_batch(raw_batch.data, LAYOUT)
    for kind in data.node_types:
        assert torch.equal(data[kind].x, training[kind].x)


@pytest.mark.parametrize("device", DEVICES)
def test_a_batch_is_encoded_on_the_device_asked(raw_batch: Batch, device: str) -> None:
    """The raw batch stays on the host, which holds float64 on every device; its encoding
    lands on the device asked and agrees with the host's."""
    on_host = encode_batch(raw_batch.data, LAYOUT)
    there = encode_batch(raw_batch.data, LAYOUT, device)
    for kind in on_host.node_types:
        assert there[kind].x.device.type == device
        torch.testing.assert_close(there[kind].x.cpu(), on_host[kind].x, atol=1e-5, rtol=0)
    assert there[PAYS].edge_index.device.type == device
    assert there.scored_index.device.type == device and there.label.device.type == device


def test_rows_not_of_their_layout_are_refused(raw_batch: Batch) -> None:
    with pytest.raises(ValueError, match="not \\(n, 23\\)"):
        encode(torch.zeros((2, 22), dtype=torch.float64), PAYMENT_LAYOUT)
    with pytest.raises(TypeError, match="float64"):
        encode(torch.zeros((2, 23), dtype=torch.float32), PAYMENT_LAYOUT)
    data = raw_batch.data.clone()
    del data[reverse(PAYS)]
    with pytest.raises(ValueError, match="not the layout's"):
        encode_batch(data, LAYOUT)
    data = raw_batch.data.clone()
    data[OWNED_BY].edge_attr = torch.zeros((data[OWNED_BY].edge_index.shape[1], 1))
    with pytest.raises(ValueError, match="owned_by carries attributes"):
        encode_batch(data, LAYOUT)


def test_an_empty_row_set_encodes_to_no_rows() -> None:
    for layout in (PAYMENT_LAYOUT, LAYOUT.nodes["customer"]):
        found = encode(torch.zeros((0, layout.raw_width), dtype=torch.float64), layout)
        assert found.shape == (0, layout.width)


def test_the_fourier_basis_resolves_the_durations_the_design_names() -> None:
    """docs/design.md, "Encoding": u = log1p(seconds) / log1p(10 years) is 0.035 for a second,
    0.21 for a minute, 0.42 for an hour, 0.58 for a day and 0.81 for 90 days. The lowest
    frequency, an eighth of a cycle per unit of u, turns less than half a circle over [0, 1],
    so its sine and cosine give u back, and the five encodings lie far apart."""
    durations = {
        SECOND_MS: 0.035,
        60 * SECOND_MS: 0.21,
        3_600 * SECOND_MS: 0.42,
        DAY_MS: 0.58,
        90 * DAY_MS: 0.81,
    }
    encodings: list[np.ndarray] = []
    for duration_ms, u in durations.items():
        assert math.log1p(duration_ms / 1000) / SCALE == pytest.approx(u, abs=0.005)
        encoded = encode_reference(np.array(duration_ms))
        angle = math.atan2(encoded[0], encoded[WIDTH // 2])
        assert angle / (2 * math.pi * 0.125) == pytest.approx(u, abs=0.005)
        encodings.append(encoded)
    for first in range(len(encodings)):
        for second in range(first + 1, len(encodings)):
            assert np.linalg.norm(encodings[first] - encodings[second]) > 1
