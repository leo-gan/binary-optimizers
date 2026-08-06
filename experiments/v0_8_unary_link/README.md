# v0.8 — Unary Swarm (link value + XOR)

Implementation of the **intended** Unary Swarm model:

- swarm of \(\pm 1\) **weights** per **link**
- **link value** from sum encoder
- per-link SGD/Adam → \(\Delta\) → decoder → **XOR** writeback

Terminology: [`docs/UNARY_SWARM_TERMINOLOGY.md`](../../docs/UNARY_SWARM_TERMINOLOGY.md)  
Plan: [`docs/UNARY_SWARM_EXPERIMENT_PLAN.md`](../../docs/UNARY_SWARM_EXPERIMENT_PLAN.md)  
Protocol: [`PROTOCOL.md`](PROTOCOL.md)

## Quick start

```bash
# Unit tests (no dataset)
pytest experiments/v0_8_unary_link/test_v0_8_unary_link.py -q

# Existence cell (WP-U1 defaults)
python experiments/v0_8_unary_link/train.py --ln-mode none --seed 42
```

## Layout

| File | Role |
|------|------|
| `layers.py` | `UnaryLinkLinear` — swarm buffer + encoders |
| `optimizer.py` | `UnaryLinkOptimizer` — SGD/Adam + decode + XOR |
| `model.py` | MNIST MLP |
| `train.py` | CLI training |
| `metrics.py` | swarm / link-value stats |
| `test_v0_8_unary_link.py` | CI unit tests |

Core modules are imported by `v0_9`–`v0_12` atlas experiments.
