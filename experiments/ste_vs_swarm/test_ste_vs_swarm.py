"""Sanity tests for ste_vs_swarm (run: pytest experiments/ste_vs_swarm/test_ste_vs_swarm.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_THIS))

from binary_optimizers.optimizers.ste import STEOptimizer
from binary_optimizers.store import build_report, connect, init_db, record_completed_run
from ste_model import BitNetSTEMLP


def test_ste_model_forward_backward_ln_modes():
    for mode in ("none", "no_affine", "affine"):
        m = BitNetSTEMLP(hidden_dim=32, ln_mode=mode)
        x = torch.randn(4, 1, 28, 28)
        y = torch.tensor([0, 1, 2, 3])
        opt = STEOptimizer(m.ste_parameters(), lr=0.1, momentum=0.9)
        ln_params = m.ln_parameters()
        ln_opt = torch.optim.SGD(ln_params, lr=1e-2) if ln_params else None
        opt.zero_grad()
        if ln_opt:
            ln_opt.zero_grad()
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        if ln_opt:
            ln_opt.step()
        out = m(x)
        assert out.shape == (4, 10)
        if mode == "affine":
            assert len(ln_params) > 0
        else:
            assert len(ln_params) == 0


def test_load_swarm_stacks():
    from train import _load_swarm_stack

    M3, O3, s3 = _load_swarm_stack("v0_3")
    m3 = M3(hidden_dim=16, n_bits=4, ln_mode="none")
    assert m3(torch.randn(2, 1, 28, 28)).shape == (2, 10)
    M4, O4, s4 = _load_swarm_stack("v0_4")
    m4 = M4(hidden_dim=16, n_trits=4, ln_mode="none")
    assert m4(torch.randn(2, 1, 28, 28)).shape == (2, 10)
    assert callable(s3) and callable(s4)


def test_report_ste_vs_swarm_section(tmp_path: Path):
    db = tmp_path / "t.duckdb"
    conn = connect(db)
    init_db(conn)
    for method, ln, acc in [
        ("ste_sgd", "none", 0.90),
        ("ste_sgd", "affine", 0.92),
        ("swarm_v0_3", "none", 0.95),
        ("swarm_v0_3", "affine", 0.97),
    ]:
        record_completed_run(
            conn,
            experiment="ste_vs_swarm",
            name=f"{method}_ln_{ln}",
            config={"method": method, "ln_mode": ln, "protocol": "ste_vs_swarm_v1"},
            history=[{"epoch": 1, "test_acc": acc}],
            seed=42,
            best_test_acc=acc,
            best_epoch=1,
            wall_sec=10.0,
            summary={"epochs_ran": 1},
        )
    text = build_report(conn)
    assert "STE vs Swarm" in text
    assert "STE (SGD+clamp)" in text or "ste_sgd" in text
    assert "0.9700" in text or "0.97" in text
    assert "Swarm − STE" in text or "swarm_v0_3" in text
    conn.close()
