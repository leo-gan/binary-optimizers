# Experiments

Research runs for **fully binary / ternary training**: discrete **networks** and
discrete **optimizers** (latent-free). See the project [README](../README.md),
[docs/PLAN.md](../docs/PLAN.md), and the Swarm explainer
[docs/SWARM_OPTIMIZER.md](../docs/SWARM_OPTIMIZER.md).

## Ladder

| Path | Role |
|------|------|
| `v0_1` | Unary ±1 Swarm (existence) |
| `v0_2` | Place-value binary bits |
| `v0_3` | Carry-safe integer register (default scaffold) |
| `v0_4` | Balanced ternary place-value |
| `v0_5_width_register` / `v0_5_width_unary` | WP1 width atlas |
| `v0_6_encoding` | WP2 encoding atlas (MNIST) |
| `v0_7_cifar_encoding` | WP3 sparse CIFAR fixed vs exp/mant |
| `ste_vs_swarm` | Optional STE comparison (not the main goal) |

Each package has `PROTOCOL.md`, `train.py`, and usually tests. Shared helpers:
`_width_atlas_common.py`. Notes: `v0_5_NOTES.md`, `v0_6_NOTES.md`, `v0_7_NOTES.md`.

## Protocol

Default train budget is **pure wall-clock** (`docs/TRAIN_BUDGET.md`). Re-runs under
a new protocol get new experiment IDs (`docs/EXPERIMENT_VERSIONS.md`).
