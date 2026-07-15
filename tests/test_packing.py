"""Tests for bit-packing round-trips."""

import math

import pytest
import torch

from binary_optimizers.inference.packing import (
    pack_activations,
    pack_binary_weights,
    packed_nbytes,
    unpack_binary_weights,
)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 8),
        (3, 7),  # not multiple of 8
        (16, 1),
        (1, 15),
        (32, 128),
        (5, 9),
    ],
)
def test_pack_unpack_roundtrip(shape):
    out_f, in_f = shape
    # Random ±1 weights
    w = torch.randint(0, 2, (out_f, in_f)).float() * 2 - 1
    packed, recovered_in = pack_binary_weights(w)
    assert recovered_in == in_f
    assert packed.dtype == torch.uint8
    assert packed.shape == (out_f, packed_nbytes(in_f))
    w_back = unpack_binary_weights(packed, in_f)
    assert w_back.shape == w.shape
    assert torch.equal(w_back, w)


def test_pack_zeros_as_plus_one():
    w = torch.zeros(2, 8)
    packed, in_f = pack_binary_weights(w)
    w_back = unpack_binary_weights(packed, in_f)
    assert torch.all(w_back == 1.0)


def test_pack_activations_shape():
    x = torch.randn(10, 13)
    packed = pack_activations(x)
    assert packed.dtype == torch.uint8
    assert packed.shape == (10, math.ceil(13 / 8))


def test_pack_all_minus_one():
    w = -torch.ones(2, 16)
    packed, in_f = pack_binary_weights(w)
    assert torch.all(packed == 0)
    assert torch.equal(unpack_binary_weights(packed, in_f), w)


def test_pack_all_plus_one():
    w = torch.ones(2, 16)
    packed, in_f = pack_binary_weights(w)
    # 8 bits set => 0xFF per byte
    assert torch.all(packed == 0xFF)
    assert torch.equal(unpack_binary_weights(packed, in_f), w)


def test_unpack_rejects_bad_width():
    packed = torch.zeros(2, 2, dtype=torch.uint8)
    with pytest.raises(ValueError):
        unpack_binary_weights(packed, in_features=8)  # expects 1 byte
