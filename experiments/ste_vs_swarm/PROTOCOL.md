# STE vs Swarm — shared-protocol comparison

**Path:** `experiments/ste_vs_swarm/`  
**Store id:** `ste_vs_swarm`  
**Dataset:** MNIST only

## Claim

Compare **straight-through estimator (STE)** binary training against **latent-free Swarm**
variants (v0.3 carry-safe place-value, v0.4 balanced ternary) under a **matched** protocol
so accuracy and wall-time differences are attributable to method, not architecture knobs.

## Shared protocol (defaults)

| Knob | Value |
|------|--------|
| Topology | Flatten → [LN] → Linear(784→H) → ReLU → [LN] → Linear(H→10) |
| `hidden` | 128 |
| activation | ReLU |
| batch size | 128 |
| seed | 42 |
| max epochs | 80 |
| early stop | patience 10, min_delta 5e-4 on **test** acc |
| LN modes | `none` \| `no_affine` \| `affine` (same meanings as v0.x) |

## Methods

| `method` | Weight state | Update |
|----------|--------------|--------|
| `ste_sgd` | FP latent `W`, forward `sign(W)` STE (`BitLinearSTE`) | SGD + momentum; weights clamped to [-1,1] (`STEOptimizer`) |
| `swarm_v0_3` | int8 bits, carry-safe place-value (v0.3) | Swarm integer ±1 steps + optional LN SGD |
| `swarm_v0_4` | int8 trits {-1,0,1}, balanced ternary (v0.4) | Swarm ternary carry-safe steps + optional LN SGD |

Method-specific defaults: `n_bits=16` (v0.3), `n_trits=10` (v0.4), STE `lr=0.1`, `momentum=0.9`, `ln_lr=1e-2`.

## Non-claims

- Not claiming identical compute or bit-exact hardware mapping.
- STE still uses FP autograd through the latent; Swarm uses buffer populations + manual STE on decode.
- Package `benchmark-mnist` (BatchNorm bit-MLP) is **out of protocol** — do not mix those numbers here.

## Logging

Each run is stored in DuckDB as:

- `experiment = ste_vs_swarm`
- `name = {method}_ln_{ln_mode}`
- config includes method, ln_mode, and hyperparameters

```bash
python experiments/ste_vs_swarm/train.py
python experiments/ste_vs_swarm/train.py --methods ste_sgd,swarm_v0_3 --ln-mode affine
./scripts/report.sh   # includes STE vs Swarm section when data present
```
