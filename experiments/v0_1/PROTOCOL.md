# Experiment v0.1 — Latent-free binary Swarm (BitNet-style MLP)

**Path:** `experiments/v0_1/` · **Results:** `results/v0_1/`

## Claim

Train a small **BitNet-style binary MLP** on **MNIST** with:

- **Weight state:** binary agents only, stored as **`int8` ∈ {-1, +1}** (design B).
- **No FP master / latent weight** \(W \in \mathbb{R}\).
- **Updates:** Swarm flips only on agents.
- **Gradient path:** **manual STE** through majority/sum decode (STE ≠ Swarm).
- **Dataset:** MNIST only.
- **LayerNorm ablations:** all three modes below.

## Non-claims

- Not bit-only backward (autograd remains FP).
- Not ternary / BitNet b1.58 product training.
- Not exponential place-value encoding (later version).
- Activations and LayerNorm arithmetic remain FP.

## LayerNorm modes (`ln_mode`)

| ID | CLI | Description |
|----|-----|-------------|
| LN0 | `none` | No LayerNorm |
| LN1 | `no_affine` | `LayerNorm(affine=False)` — FP mean/var only, no γ,β |
| LN2 | `affine` | `LayerNorm(affine=True)` — γ,β via small FP SGD (norm crutch) |

## Architecture

```
Flatten
→ [optional LN on 784]
→ Int8SwarmLinear(784 → H)     # majority decode, manual STE
→ SquaredReLU
→ [optional LN on H]
→ Int8SwarmLinear(H → 10)
→ CrossEntropy
```

Defaults: `H=128`, `swarm_size=32`, no bias on swarm linears.

## Optimizer

- **Swarm** on all `int8` populations: stochastic flips toward \(-\mathrm{sign}(\mathrm{EMA}[g])\).
- Flip probability \(\min(p_{\max}, |EMA[g]| \cdot r)\) with defaults \(r=10^4\), \(p_{\max}=0.15\), EMA momentum \(0.9\).
- **No FP master weights**; optional FP state is only the per-weight EMA pressure (and LN γ,β in LN2).
- **LN2 only:** separate SGD group for LayerNorm γ,β (`ln_lr`).
- Hidden activation default: **ReLU** (stable); `--activation squared_relu` for BitNet-faithful FFN.

## Success criteria

- Test accuracy rises and plateaus (early-stop on test acc).
- After every step: `population ∈ {-1, +1}` int8.
- Primary “pure” reports: LN0 and LN1; LN2 reported as hybrid.

## Results (seed=42, defaults, early-stop patience=10)

| ln_mode | best test acc | best epoch | epochs ran |
|---------|---------------|------------|------------|
| `none` | **0.9239** | 19 | 29 |
| `no_affine` | 0.9166 | 3 | 13 |
| `affine` | 0.9188 | 1 | 11 |

Artifacts: `results/v0_1/`, `checkpoints/v0_1/`.

Notes: pure `none` was most stable long-term; LN modes peak early then degrade under continued flips (best checkpoint retained).

## Naming

Human label **v0.1** · code/results paths **`v0_1`**. Next lines: `v0_2`, `v1_0`, …
