# Experiment v0.2 — Place-value (exponential) binary Swarm

**Path:** `experiments/v0_2/` · **Results:** `results/v0_2/`  
**Baseline:** v0.1 unary majority Swarm (`experiments/v0_1/`)

## Claim

Same latent-free setup as v0.1 (int8 agents, no FP master weight, Swarm flips
only), but agents use **place-value / exponential** coding instead of unary
majority:

- Agent \(i\) contributes \(\pm 2^{i}\) (bit plane \(i\)).
- Weight decode \(s_{\mathrm{norm}} = \sum_i a_i\,2^{i} / \sum_j 2^{j} \in [-1,1]\).
- Forward uses **multi-level** \(s_{\mathrm{norm}}\) (not hard majority sign), so an
  LSB flip is a small step and an MSB flip is a large one.
- Flip schedule: **LSB-easier** — \(\mathrm{prob} \propto |EMA[g]| \cdot r \cdot 2^{-i}\).

This tests the research idea: swarm as **exponential FP-like dynamic range**,
not an integer thermometer count. Storage is still pure binary agents.

## Non-claims

- Not true binary carry arithmetic on every update (independent bit flips with
  LSB bias; carry-safe rules are a later version if needed).
- Not ternary; not CIFAR / Transformer.
- Autograd / activations / LN math still FP.

## LayerNorm modes

Same three as v0.1: `none` | `no_affine` | `affine`.

## Architecture

Identical MLP to v0.1 (BitNet-style: Flatten → [LN?] → PlaceValueSwarmLinear → ReLU
→ [LN?] → PlaceValueSwarmLinear), default `hidden=128`, `n_bits=16`.

## Success criteria

- Test acc competitive with v0.1 (~92% on MNIST with `ln=none`).
- Prefer fewer bits (`n_bits ≤ 16`) vs v0.1 `swarm_size=32` unary when possible.
- int8 ±1 invariant after every step.

## Results (seed=42, n_bits=16, early-stop patience=10)

| ln_mode | best test | best epoch | vs v0.1 none (0.9239) |
|---------|-----------|------------|------------------------|
| `none` | **0.9289** | 26 | +0.5 pp |
| `no_affine` | **0.9357** | 5 | +1.2 pp |
| `affine` | **0.9365** | 5 | +1.3 pp |

Place-value with **16 bits** matches/beats unary **32 agents**, with clear LSB > mid > MSB flip rates.

Artifacts: `results/v0_2/`, `checkpoints/v0_2/`.

## Naming

**v0.2** human · **`v0_2`** paths. Next: carry-safe updates / ternary / smaller n_bits ablations.
