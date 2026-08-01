# Experiment v0.3 — Carry-safe place-value Swarm

**Path:** `experiments/v0_3/` · **Results:** `results/v0_3/`  
**Baseline:** v0.2 independent bit flips (`experiments/v0_2/`)

## Claim

Keep latent-free **binary place-value** weights, but update via a **carry-safe
integer register** instead of independent per-bit flips:

- Bits \(b_i \in \{0,1\}\) encode integer \(v = \sum_i b_i 2^{i} \in [0, 2^{n}-1]\).
- Weight \(w = 2v/(2^{n}-1) - 1 \in [-1,1]\).
- Optimizer: EMA of \(\partial L/\partial w\); with probability \(\propto |\mathrm{EMA}|\),
  apply \(v \leftarrow \mathrm{clip}(v \pm 1)\) (direction \(-\mathrm{sign}(g)\)).
- Re-encode \(v\) to bits — **carries are exact** (true binary ±1 steps).

## Non-claims

- Not ternary (→ v0.4).
- Not bit-only backward; LN/acts still FP.

## LayerNorm modes

`none` | `no_affine` | `affine` (same as v0.1/v0.2).

## Success criteria

- Match or beat v0.2 (`none` 0.9289, best LN ~0.936).
- Smoother late training (fewer random MSB flips).
- Bits always \{0,1\} after each step.

## Results (seed=42, n_bits=16, adaptive Δv ≤ 512)

| ln_mode | best test | best epoch |
|---------|-----------|------------|
| `none` | **0.9589** | 64 |
| `no_affine` | **0.9723** | 17 |
| `affine` | **0.9745** | 27 |

Beats v0.2 substantially (+3–4 pp). Carry-safe integer steps are the win.

## Naming

**v0.3** · paths **`v0_3`**.
