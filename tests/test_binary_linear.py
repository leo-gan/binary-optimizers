"""Tests for XNOR+popcount binary linear vs F.linear(sign(x), sign(w))."""
import pytest

import torch
import torch.nn.functional as F

from binary_optimizers.inference.binary_linear import (
    BinaryLinear,
    binary_linear_forward,
    reference_binary_linear,
)
from binary_optimizers.inference.extract import extract_packed_weights
from binary_optimizers.inference.packing import pack_binary_weights
from binary_optimizers.models.bit_layers import BitLinearSTE
from binary_optimizers.models.mnist import create_mnist_bit_mlp, create_mnist_swarm_mlp


def _sign_pm1(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t >= 0, torch.ones_like(t), -torch.ones_like(t))


@pytest.mark.parametrize("batch,in_f,out_f", [(1, 8, 4), (7, 15, 3), (16, 128, 10), (4, 784, 32)])
def test_binary_linear_matches_reference(batch, in_f, out_f):
    torch.manual_seed(0)
    x = torch.randn(batch, in_f)
    w = torch.randint(0, 2, (out_f, in_f)).float() * 2 - 1
    packed, recovered = pack_binary_weights(w)
    assert recovered == in_f

    out_bin = binary_linear_forward(x, packed, in_f)
    out_ref = reference_binary_linear(x, w)
    out_flin = F.linear(_sign_pm1(x), _sign_pm1(w))

    assert torch.allclose(out_bin, out_ref)
    assert torch.allclose(out_bin, out_flin)
    # Exact integer-valued outputs
    assert torch.equal(out_bin, out_bin.round())


def test_binary_linear_with_bias():
    torch.manual_seed(1)
    x = torch.randn(5, 12)
    w = torch.randint(0, 2, (6, 12)).float() * 2 - 1
    bias = torch.randn(6)
    packed, in_f = pack_binary_weights(w)
    out = binary_linear_forward(x, packed, in_f, bias=bias)
    ref = reference_binary_linear(x, w, bias=bias)
    assert torch.allclose(out, ref)


def test_binary_linear_module():
    torch.manual_seed(2)
    w = torch.randn(10, 20)
    layer = BinaryLinear.from_weight(w)
    x = torch.randn(3, 20)
    out = layer(x)
    ref = reference_binary_linear(x, w)
    assert torch.allclose(out, ref)
    assert torch.allclose(layer.unpacked_weight(), _sign_pm1(w))


def test_no_float_matmul_path_uses_uint8_weights():
    w = torch.ones(4, 16)
    packed, in_f = pack_binary_weights(w)
    assert packed.dtype == torch.uint8
    x = torch.ones(2, 16)
    out = binary_linear_forward(x, packed, in_f)
    # all +1: each output = in_features
    assert torch.allclose(out, torch.full((2, 4), 16.0))


def test_extract_from_bit_mlp():
    model = create_mnist_bit_mlp(hidden_dim=32)
    packed = extract_packed_weights(model)
    assert len(packed.layers) >= 2
    assert packed.total_packed_bytes() > 0
    assert packed.total_weight_bits() == sum(
        L.in_features * L.out_features for L in packed.layers
    )


def test_extract_swarm_majority_vs_full():
    model = create_mnist_swarm_mlp(hidden_dim=16, swarm_size=8)
    maj = extract_packed_weights(model, use_swarm_majority=True)
    full = extract_packed_weights(model, use_swarm_majority=False)
    assert maj.total_weight_bits() * 8 == full.total_weight_bits()
    assert full.total_packed_bytes() > maj.total_packed_bytes()


def test_bitlinear_ste_extract_and_forward():
    torch.manual_seed(3)
    layer = BitLinearSTE(20, 5, bias=True)
    packed = extract_packed_weights(layer)
    assert len(packed.layers) == 1
    L = packed.layers[0]
    x = torch.randn(4, 20)
    out = binary_linear_forward(x, L.packed_w, L.in_features, L.bias)
    ref = F.linear(_sign_pm1(x), _sign_pm1(layer.weight.data), layer.bias.data)
    assert torch.allclose(out, ref)
