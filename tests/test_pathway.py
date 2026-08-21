"""Unit tests for assembly-index proxy and PathwayOptimizer (no dataset)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from binary_optimizers.assembly import (
    approximate_assembly_index,
    mean_assembly_index,
    pack_signs,
    snap_binary_,
)
from binary_optimizers.optimizers.pathway import PathwayOptimizer, _unravel


def test_pack_signs_length_is_ceil_n_over_8():
    t = torch.ones(10)
    packed = pack_signs(t)
    assert len(packed) == math.ceil(10 / 8)
    t2 = torch.ones(16)
    assert len(pack_signs(t2)) == 2


def test_structured_tensor_has_lower_assembly_index_than_random():
    g = torch.Generator().manual_seed(0)
    row = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    structured = row.repeat(256, 32)  # 256 x 256 tiled
    random = torch.randint(0, 2, (256, 256), generator=g).float().mul_(2).sub_(1)
    a_s = approximate_assembly_index(structured)
    a_r = approximate_assembly_index(random)
    assert a_s < a_r
    assert a_s < 0.5
    assert a_r > 0.8


def test_mean_assembly_index_empty_and_weighted():
    assert mean_assembly_index([]) == 0.0
    t = torch.ones(32, 32)
    assert mean_assembly_index([t]) == approximate_assembly_index(t)


def test_snap_binary_maps_zero_to_plus_one():
    t = torch.tensor([[-0.3, 0.0, 2.0]])
    snap_binary_(t)
    assert torch.equal(t, torch.tensor([[-1.0, 1.0, 1.0]]))


def test_unravel_c_order():
    shape = torch.Size((3, 4, 5))
    assert _unravel(0, shape) == (0, 0, 0)
    assert _unravel(5, shape) == (0, 1, 0)
    assert _unravel(20, shape) == (1, 0, 0)
    assert _unravel(59, shape) == (2, 3, 4)


def test_step_requires_closure():
    w = nn.Parameter(torch.ones(2, 2))
    opt = PathwayOptimizer([w], candidates=1, lambda_a=0.0, ranking="random")
    with pytest.raises(TypeError, match="closure"):
        opt.step()


def test_step_preserves_pm1():
    torch.manual_seed(1)
    w = nn.Parameter(torch.randn(4, 5))
    opt = PathwayOptimizer([w], candidates=3, lambda_a=0.0, ranking="random")
    x = torch.randn(6, 5)
    y = torch.randn(6, 4)

    def closure():
        opt.zero_grad()
        pred = F.linear(x, w)
        loss = (pred - y).pow(2).mean()
        loss.backward()
        return loss, pred

    opt.step(closure)
    uniq = set(w.detach().unique().tolist())
    assert uniq.issubset({-1.0, 1.0})
    assert opt.last_proposed == 3


def test_reject_flip_that_increases_loss():
    # y = w x; w=+1, x=1, target=1 → already optimal.
    w = nn.Parameter(torch.ones(1, 1))
    opt = PathwayOptimizer([w], candidates=1, lambda_a=0.0, ranking="random")
    x = torch.ones(1, 1)
    target = torch.ones(1, 1)

    def closure():
        opt.zero_grad()
        pred = F.linear(x, w)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        return loss, pred

    opt.step(closure)
    assert torch.equal(w.data, torch.ones(1, 1))
    assert opt.last_accepted == 0
    assert opt.last_proposed == 1


def test_accept_flip_that_fixes_wrong_sign():
    # w starts at -1; target wants +1.
    w = nn.Parameter(-torch.ones(1, 1))
    opt = PathwayOptimizer([w], candidates=1, lambda_a=0.0, ranking="random")
    x = torch.ones(1, 1)
    target = torch.ones(1, 1)

    def closure():
        opt.zero_grad()
        pred = F.linear(x, w)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        return loss, pred

    opt.step(closure)
    assert torch.equal(w.data, torch.ones(1, 1))
    assert opt.last_accepted == 1
    assert opt.last_delta_loss < 0.0


def test_one_d_params_are_not_flipped():
    torch.manual_seed(2)
    w = nn.Parameter(torch.ones(2, 2))
    b = nn.Parameter(torch.tensor([0.25, -0.5, 0.75]))
    before_b = b.data.clone()
    opt = PathwayOptimizer(
        [w, b],
        candidates=4,
        lambda_a=0.0,
        ranking="random",
        bn_lr=0.0,
    )
    x = torch.randn(3, 2)
    y = torch.randn(3, 2)

    def closure():
        opt.zero_grad()
        pred = F.linear(x, w)
        loss = (pred - y).pow(2).mean()
        loss.backward()
        return loss, pred

    opt.step(closure)
    assert torch.equal(b.data, before_b)
    uniq = set(w.detach().unique().tolist())
    assert uniq.issubset({-1.0, 1.0})


def test_hybrid_only_proposes_ste_desired_flips():
    # w=+1, target=-1 → STE wants the flip and loss drops if accepted.
    w = nn.Parameter(torch.ones(1, 1))
    opt = PathwayOptimizer(
        [w],
        candidates=1,
        lambda_a=0.0,
        ranking="ste_topk",
        mode="hybrid",
        assembly_of="weights",
    )
    x = torch.ones(1, 1)
    target = -torch.ones(1, 1)

    def closure():
        opt.zero_grad()
        pred = F.linear(x, w)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        return loss, pred

    opt.step(closure)
    assert opt.last_proposed == 1
    assert torch.equal(w.data, -torch.ones(1, 1))
    assert opt.last_accepted == 1
