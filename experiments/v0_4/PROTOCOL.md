# Experiment v0.4 — Balanced ternary place-value Swarm

**Path:** `experiments/v0_4/` · **Results:** `results/v0_4/`  
**Baseline:** v0.2/v0.3 binary place-value

## Claim

Latent-free **ternary** place-value coding (BitNet-adjacent alphabet):

- Digits \(d_i \in \{-1,0,+1\}\), places \(3^{i}\) (balanced ternary).
- Integer range \(\bigl[-(3^{n}-1)/2,\\ (3^{n}-1)/2\bigr]\) uniquely represented.
- Weight \(w = s / s_{\max}\in[-1,1]\).
- Carry-safe update: map to non-negative base-3 index, \(\pm 1\) step, re-encode
  to balanced ternary digits (exact ternary carry).

## Non-claims

- Not full BitNet Transformer; not int8 activation quant required.
- Not bit-only backward.

## Defaults

- `n_trits=10` (≈15.8 bits of information; comparable to 16 binary bits).
- LN modes: `none` | `no_affine` | `affine`.

## Success criteria

- Competitive with v0.2/v0.3 (~93% MNIST).
- Digits stay in {-1,0,1}.

## Results (seed=42, n_trits=10, adaptive Δs ≤ 64)

| ln_mode | best test | best epoch | zero_frac (final) |
|---------|-----------|------------|-------------------|
| `none` | **0.9551** | 37 | ~0.29 |
| `no_affine` | **0.9717** | 16 | ~0.33 |
| `affine` | **0.9737** | 16 | ~0.33 |

Matches v0.3 LN modes; ternary sparsity (`zero`) emerges naturally.

## Naming

**v0.4** · paths **`v0_4`**.
