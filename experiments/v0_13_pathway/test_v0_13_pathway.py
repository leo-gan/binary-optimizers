"""Unit tests for v0_13 pathway experiment (no dataset)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_THIS))

from binary_optimizers.models.pathway_norm import PathwayNorm
from binary_optimizers.optimizers.pathway import PathwayOptimizer
from binary_optimizers.store.versions import TRAIN_BUDGET_PROTOCOL, get_meta

from compare_norms import COMPARE_NORMS, format_markdown, summarize
from model import PathwayMLP


def test_init_snap_is_pm1():
    torch.manual_seed(0)
    model = PathwayMLP(hidden_dim=8, in_dim=16, n_classes=3)
    model.assert_binary()
    for p in model.binary_parameters():
        assert p.eq(0).sum().item() == 0


def test_one_fake_batch_step_stays_binary():
    torch.manual_seed(1)
    model = PathwayMLP(hidden_dim=8, in_dim=16, n_classes=3)
    opt = PathwayOptimizer(
        model.parameters(),
        candidates=2,
        lambda_a=0.0,
        ranking="ste_topk",
        assembly_of="weights",
        mode="pathway",
    )
    x = torch.randn(4, 16)
    y = torch.tensor([0, 1, 2, 1])

    def closure():
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        return loss, logits

    opt.step(closure)
    model.assert_binary()
    assert opt.last_proposed == 2
    assert 0 <= opt.last_accepted <= 2


def test_pathway_norm_hidden_is_order_one():
    torch.manual_seed(2)
    model = PathwayMLP(hidden_dim=8, in_dim=16, n_classes=3, norm="pathway")
    model.train()
    x = torch.randn(4, 16)
    h = model.norm(model.fc1(model.flatten(x)))
    assert torch.isfinite(h).all()
    assert h.abs().mean().item() < 5.0
    stats = model.norm_stats()
    assert "agreement" in stats


def test_pathway_norm_strength_zero_constructs():
    model = PathwayMLP(hidden_dim=8, in_dim=16, n_classes=3, norm="pathway", norm_strength=0.0)
    assert isinstance(model.norm, PathwayNorm)
    assert model.norm.strength == 0.0
    h = model.hidden_preact(torch.randn(2, 16))
    assert h.shape == (2, 8)


def test_compare_norms_summary_picks_winner():
    assert COMPARE_NORMS == ("pathway", "layernorm")
    summary = summarize(
        [
            {
                "norm": "pathway",
                "run_tag": "cmp_pathway",
                "best_test_acc": 0.55,
                "best_epoch": 1,
                "final_test_acc": 0.54,
                "final_test_loss": 3.0,
                "epochs_ran": 4,
                "wall_sec": 200.0,
            },
            {
                "norm": "layernorm",
                "run_tag": "cmp_layernorm",
                "best_test_acc": 0.62,
                "best_epoch": 8,
                "final_test_acc": 0.60,
                "final_test_loss": 1.5,
                "epochs_ran": 12,
                "wall_sec": 1200.0,
            },
        ]
    )
    assert summary["winner_norm"] == "layernorm"
    md = format_markdown(summary)
    assert "`pathway`" in md and "`layernorm`" in md
    assert "0.6200" in md


def test_registry_v0_13_pathway():
    m = get_meta("v0_13_pathway")
    assert m["code_dir"] == "experiments/v0_13_pathway"
    assert m["protocol"] == TRAIN_BUDGET_PROTOCOL
    assert m["parent"] == "v0_1"
