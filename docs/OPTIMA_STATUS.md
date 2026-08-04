# Optima status

Context: the project goal is **fully binary/ternary training** (latent-free NN +
discrete optimizer). These curves support representation laws under that constraint.
See root [README.md](../README.md) and [PLAN.md](PLAN.md).

Snapshot of **parent** experiment results already on disk. Many used epoch-based
early stop (not pure wall). Use for scientific conclusions at sketch quality;
re-runs under `*_1` ids + `pure_wall_budget_v1` refine, not necessarily reverse.

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
| B encoding (MNIST) | **Yes** — fixed ≫ exp/mant | No expand required |
| B on CIFAR (sparse v0_7) | **Yes** — fixed still wins (~0.51 best) | Optional stronger CIFAR net |

Expand only for polish (register densify, n=32 rescue, conv CIFAR) or pure-wall
re-runs under `*_1` ids—not to "find" a missing peak.

### CIFAR sparse (`v0_7_cifar_encoding`)

| tag | best test |
|-----|----------:|
| **fixed_n8** | **0.5088** |
| fixed_n16 | 0.4910 |
| exp_mant2_n8 | 0.4409 |
| exp_mant2_n16 | 0.4196 |

---

## Pure-wall re-run of winners (`pure_wall_budget_v1`, seed 42)

Budget: `max_wall_sec=1200`, wall patience=150 s, no epoch early-stop.

| Cell | Legacy best | Pure-wall best | Δ | Pure-wall wall / epochs | Stop |
|------|------------:|---------------:|--:|-------------------------|------|
| Register n=8 (`v0_5_1_width_register`) | 0.9633 | **0.9641** | +0.0008 | 545 s / 65 ep (best@48) | patience_wall |
| Encoding fixed@8 (`v0_6_1_encoding`) | 0.9633 | **0.9641** | +0.0008 | 485 s / 71 ep (best@48) | patience_wall |
| Unary S=256 (`v0_5_1` / `v0_6_1`) | 0.9617 / 0.9582 | 0.9553 | −0.006 / −0.003 | ~700–800 s / **6 ep** (best@4) | patience_wall |

### Takeaways

1. **Fixed-point / register n=8** is stable under pure wall; slightly better than
   legacy (more epochs allowed before 150 s stall ≈ 20+ short epochs).  
2. **Unary S=256 under 150 s wall patience is harsh:** one epoch ≈ 110 s, so stall
   fires after **~1–2 non-improving epochs**. Legacy epoch patience (10) ran ~26
   epochs and reached higher best. Pure wall is fairer *across* cells but **tight
   for slow epochs** if patience is a fixed fraction of wall.  
3. Ranking under pure wall still has **fixed_n8 > unary_S256** (0.964 vs 0.955),
   same order as legacy.

### Optional follow-up (not run)

- Unary-only: `--patience-frac 0.25` (300 s stall ≈ 2–3 unary epochs) or longer
  `--max-wall-sec` for polish without changing the pure-wall *principle*.
