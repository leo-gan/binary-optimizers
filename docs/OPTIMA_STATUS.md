# Optima status (legacy runs; epoch-based protocol)

Snapshot of **parent** experiment results already on disk. These used epoch-based
early stop (not pure wall). Use for scientific conclusions at sketch quality;
re-runs under `*_1` ids + `pure_wall_budget_v1` would refine, not necessarily reverse.

## Register width (`v0_5_width_register`)

| n_bits | best test | shape |
|-------:|----------:|-------|
| 1 | 0.679 | underfit |
| 2 | 0.880 | rising |
| 3 | 0.913 | rising |
| 4 | 0.945 | rising |
| 6 | 0.959 | rising |
| **8** | **0.963** | **peak** |
| 16 | 0.959 | slight drop |
| 32–62 | ~0.89 | cliff / plateau |

**Optimum:** **n ≈ 8** (local, unimodal). Neighbors 6 and 16 are close but lower.
Optional densify {7,9,10,12} is polish only—not required to claim a peak region.

## Unary population (`v0_5_width_unary`)

| S | best test | Δ vs prev |
|--:|----------:|----------:|
| 8 | 0.860 | — |
| 16 | 0.893 | + |
| 32 | 0.924 | + |
| 64 | 0.940 | + |
| 128 | 0.957 | + |
| 256 | 0.962 | + |
| 512 | 0.959 | ≈ flat / noise |
| **1024** | **0.964** | ≈ flat |

**Optimum:** **soft plateau S ≈ 256–1024** (practical pick **S=256** for cost;
peak sample S=1024). No need for S=2048 under current noise level.

## Encoding class (`v0_6_encoding`)

| class | best | note |
|-------|-----:|------|
| **fixed n=8** | **0.963** | winner |
| unary S=256 | 0.958 | strong baseline |
| fixed n=16 | 0.954 | 2nd structure |
| block_scale n=8 | 0.951 | behind fixed |
| exp_mant * | 0.89–0.92 | worse with more n_e |

**Optimum (encoding class):** **pure fixed-point** at n=8 on this net.
Float-like exp/mant splits did not help. Optional: n=32 rescue probe (not run)
to test whether exp/mant helps only when fixed-point cliffs—not required to
rank classes at healthy budgets.

## Conclusion

| Question | Have a usable optimum? | Action |
|----------|------------------------|--------|
| A register n | **Yes** — n≈8 | No expand required |
| A unary S | **Yes** — plateau 256–1024 | No expand required |
| B encoding | **Yes** — fixed ≫ exp_mant | No expand required |

Expand only for polish (register densify, n=32 rescue) or pure-wall re-runs of
winners under `*_1` ids—not to “find” a missing peak.
