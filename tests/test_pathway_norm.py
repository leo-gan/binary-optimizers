"""Unit tests for PathwayNorm (no dataset)."""

from __future__ import annotations

import torch

from binary_optimizers.models.pathway_norm import PathwayNorm, _sign_pm1


def test_huge_input_becomes_order_one():
    torch.manual_seed(0)
    layer = PathwayNorm(16, momentum=1.0)
    layer.train()
    x = torch.randn(8, 16) * 200.0
    y = layer(x)
    assert y.shape == x.shape
    assert y.abs().mean().item() < 5.0
    assert torch.isfinite(y).all()


def test_per_sample_scale_eval_batch_one():
    layer = PathwayNorm(8, momentum=1.0)
    layer.train()
    layer(torch.randn(4, 8) * 10)
    layer.eval()
    y = layer(torch.randn(1, 8) * 10)
    assert y.shape == (1, 8)
    assert torch.isfinite(y).all()


def test_constant_channel_is_centered_after_train_step():
    layer = PathwayNorm(2, momentum=1.0, strength=1.0)
    layer.train()
    # Channel 0 always +5, channel 1 always -3.
    x = torch.stack([torch.full((6,), 5.0), torch.full((6,), -3.0)], dim=1)
    _ = layer(x)
    y = layer(x)
    # Residual should be ~0 after motif_mag matches |x|.
    assert y.abs().mean().item() < 0.2


def test_eval_freezes_motif():
    layer = PathwayNorm(4, momentum=1.0)
    layer.train()
    layer(torch.ones(3, 4))
    acc = layer.motif_acc.clone()
    layer.eval()
    layer(-torch.ones(3, 4))
    assert torch.equal(layer.motif_acc, acc)


def test_binarize_outputs_pm1_and_has_grad():
    layer = PathwayNorm(5, momentum=1.0, binarize=True)
    x = torch.randn(3, 5, requires_grad=True)
    y = layer(x)
    uniq = set(y.detach().unique().tolist())
    assert uniq.issubset({-1.0, 1.0})
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_conv_nchw_shape():
    layer = PathwayNorm(3, momentum=1.0)
    x = torch.randn(2, 3, 4, 4) * 20
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_affine_params_are_one_d():
    layer = PathwayNorm(7, affine=True)
    names = {n for n, _ in layer.named_parameters()}
    assert names == {"weight", "bias"}
    assert layer.weight.shape == (7,)


def test_sign_pm1_maps_zero():
    t = torch.tensor([-2.0, 0.0, 3.0])
    assert torch.equal(_sign_pm1(t), torch.tensor([-1.0, 1.0, 1.0]))


def test_wrong_channel_count_raises():
    layer = PathwayNorm(4)
    try:
        layer(torch.randn(2, 3))
    except ValueError as exc:
        assert "channels" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_backward_through_residual():
    layer = PathwayNorm(6, momentum=0.5)
    x = (torch.randn(5, 6) * 8).requires_grad_(True)
    y = layer(x)
    y.pow(2).mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_stats_populated():
    layer = PathwayNorm(4, momentum=1.0)
    layer(torch.randn(3, 4))
    s = layer.stats()
    assert -1.0 <= s["agreement"] <= 1.0
    assert s["scale"] > 0.0
    assert 0.0 <= s["motif_frac_plus"] <= 1.0
