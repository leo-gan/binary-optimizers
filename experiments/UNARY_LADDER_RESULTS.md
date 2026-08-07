# Unary link Swarm ladder — consolidated results

**Branch:** `plan/unary-swarm`  
**Window:** 2026-08-05 → 2026-08-06  
**Hardware:** CPU only  
**Monitor:** Grok watch → `DONE` at 2026-08-06T00:19

Full ladder script: `logs/run_unary_ladder.sh`  
Master log: `logs/unary_ladder_master.log`

## Status

| WP | Experiment | Status |
|----|------------|--------|
| U1 | `v0_8_unary_link` | **done** — existence |
| U2 | `v0_9_unary_decoder` | **done** — recipe lock |
| U3 | `v0_10_unary_width` | **done** — S atlas (sgd) |
| U4 | `v0_11_unary_encoder` | **done** — encoder rank (sgd) |
| U5 | `v0_12_unary_cifar` | **done** — sparse CIFAR (budget-confounded) |

Per-WP detail: `v0_8_NOTES.md` … `v0_12_NOTES.md`.

## Headline numbers (seed 42)

| WP | Best cell | Best test |
|----|-----------|----------:|
| U1 existence | S=256 fixed sgd density | 0.8988 |
| U2 recipe | **adam + density** | **0.9366** |
| U2 #2 | sgd_m + density | 0.9335 |
| U3 width | **S=128** (sgd+fixed) | **0.9000** |
| U4 encoder | **majority** (sgd) | **0.9377** |
| U5 CIFAR | S=64 fixed sgd | 0.3813 |

## Locked defaults (post-ladder)

For **new** Unary link runs on MNIST-scale toys:

| Knob | Default |
|------|---------|
| opt | **adam** (lr=1e-3) |
| decoder | **density** |
| p_noise | **0.001** |
| S (cheap band) | **32–128** under sgd curve; re-check under adam |
| encoder | open: **majority** won under sgd; **fixed** was plan default — prefer majority unless multi-level is the object of study |

## Shared issues

1. Discrete flip rate ≈ `p_noise` (α|Δ| ineffective at default scale).  
2. U3–U5 used **sgd**, not U2 adam — fair re-runs pending.  
3. CIFAR large-S cells hit max wall with few epochs.

## Artifacts root

```
results/v0_8_unary_link/
results/v0_9_unary_decoder/
results/v0_10_unary_width/
results/v0_11_unary_encoder/
results/v0_12_unary_cifar/
checkpoints/v0_*_unary_*/
logs/v0_*.log
```
