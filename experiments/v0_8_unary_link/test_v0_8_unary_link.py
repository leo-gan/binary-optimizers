"""Unit tests for v0_8 Unary link Swarm (no dataset)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_THIS))

from layers import UnaryLinkLinear, encode_sum
from model import UnaryLinkMLP
from optimizer import UnaryLinkOptimizer


def test_encode_sum_fixed_and_majority():
    s = torch.tensor([[-4.0, 0.0, 4.0]])
    w = encode_sum(s, swarm_size=4, encoder="fixed", tanh_tau=0.0)
    assert torch.allclose(w, torch.tensor([[-1.0, 0.0, 1.0]]))
    w_m = encode_sum(s, swarm_size=4, encoder="majority", tanh_tau=0.0)
    assert w_m[0, 0].item() == -1.0
    assert w_m[0, 1].item() == 1.0  # tie → +1
    assert w_m[0, 2].item() == 1.0


def test_encode_signed_sqrt_and_tanh():
    s = torch.tensor([4.0, -4.0])
    w = encode_sum(s, swarm_size=16, encoder="signed_sqrt", tanh_tau=0.0)
    assert w[0].item() > 0 and w[1].item() < 0
    w_t = encode_sum(s, swarm_size=16, encoder="tanh", tanh_tau=8.0)
    assert w_t.abs().max().item() <= 1.0 + 1e-5


def test_forward_backward_link_grad_shape():
    layer = UnaryLinkLinear(16, 8, swarm_size=32, encoder="fixed")
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)
    y.sum().backward()
    assert layer.last_link_grad is not None
    assert layer.last_link_grad.shape == (8, 16)


def test_xor_double_flip_identity():
    layer = UnaryLinkLinear(4, 2, swarm_size=8)
    before = layer.swarm.clone()
    # Flip all bits twice via manual XOR-equivalent
    layer.swarm.mul_(-1)
    layer.swarm.mul_(-1)
    assert torch.equal(layer.swarm, before)


def test_optimizer_step_preserves_pm1_and_can_flip():
    torch.manual_seed(0)
    layer = UnaryLinkLinear(8, 4, swarm_size=16, encoder="fixed")
    opt = UnaryLinkOptimizer([layer], opt="sgd", lr=1.0, alpha=100.0, p_max=0.5)
    x = torch.randn(8, 8)
    opt.zero_grad()
    loss = layer(x).pow(2).mean()
    loss.backward()
    before = layer.swarm.clone()
    flip = opt.step()
    layer.enforce_binary_()
    uniq = set(layer.swarm.unique().tolist())
    assert uniq.issubset({-1, 1})
    assert 0.0 <= flip <= 1.0
    # With large alpha/lr, expect some flips most of the time
    assert flip > 0.0 or torch.equal(before, layer.swarm)


def test_freeze_swarm_baseline():
    torch.manual_seed(1)
    layer = UnaryLinkLinear(8, 4, swarm_size=8)
    opt = UnaryLinkOptimizer([layer], freeze_swarm=True, lr=1.0, alpha=100.0)
    before = layer.swarm.clone()
    x = torch.randn(4, 8)
    opt.zero_grad()
    layer(x).sum().backward()
    opt.step()
    assert torch.equal(layer.swarm, before)


def test_model_train_step_all_opts_and_decoders():
    for opt_name in ("sgd", "sgd_m", "adam"):
        for dec in ("density", "thresholded", "sign_noise"):
            m = UnaryLinkMLP(hidden_dim=16, swarm_size=8, ln_mode="none")
            opt = UnaryLinkOptimizer(
                m.swarm_layers(), opt=opt_name, decoder=dec, lr=0.1
            )
            x = torch.randn(2, 1, 28, 28)
            y = torch.tensor([0, 1])
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            m.assert_binary_invariants()


def test_encoders_on_model():
    for enc in ("fixed", "tanh", "signed_sqrt", "majority"):
        m = UnaryLinkMLP(hidden_dim=8, swarm_size=8, encoder=enc)
        out = m(torch.randn(2, 1, 28, 28))
        assert out.shape == (2, 10)
