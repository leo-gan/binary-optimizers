# Experiment v0.7 — CIFAR-10 encoding probe (WP3 sparse)

**Path:** `experiments/v0_7_cifar_encoding/`  
**Run / DB id:** `v0_7_cifar_encoding`  
**Results:** `results/v0_7_cifar_encoding/`  
**Plan:** WP3 — does fixed-point still beat exp/mant when data is harder?

## Claim

Minimal comparison: **fixed-point vs exp_mant (n_e=2)** at \(n\in\{8,16\}\) on
CIFAR-10 with a flat BitNet-style MLP (3072→128→10). Same train recipe; pure wall
budget (option B: 40 min wall).

## Cells (exactly 4)

| tag | encoding |
|-----|----------|
| `fixed_n8` | pure place-value, n=8 |
| `fixed_n16` | pure place-value, n=16 |
| `exp_mant2_n8` | n_e=2, n_m=6 |
| `exp_mant2_n16` | n_e=2, n_m=14 |

## Protocol

| Knob | Value |
|------|--------|
| Data | CIFAR-10 (normalize 0.5) |
| Net | Flatten → Linear(3072→128) → ReLU → Linear(128→10) |
| `ln_mode` | `none` |
| seed | 42 |
| Budget | **pure wall**: `max_wall_sec=2400`, `patience_frac=0.125` (300 s stall) |
| min_delta | 0 |

No per-cell hparam search. Scaffold = v0.6 encoding layers.

## Deliverable

Ranking fixed vs exp_mant on CIFAR; compare to MNIST WP2 conclusion.
