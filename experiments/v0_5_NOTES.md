# v0.5 width atlas — notes

**Branch:** `feat/v0_5-width-atlas`  
**Question A:** How much discrete state per weight? Two families, not mixed.

| Experiment | Coding | Width axis |
|------------|--------|------------|
| `v0_5_width_register` | v0.3 carry-safe place-value | \(n_{\mathrm{bits}}\) |
| `v0_5_width_unary` | v0.1 unary majority Swarm | population \(S\) |

Protocol: MNIST MLP `hidden=128`, `ln_mode=none`, seed 42, early stop 80/10.  
**No per-width hyperparam search.** Register runs scale `max_step` / `step_scale` with
\(v_{\max}\) so \(\Delta v / v_{\max}\) matches the n=16 reference (fair width compare).

## Register curve (seed 42, ln=none)

| n_bits | best test acc | note |
|-------:|--------------:|------|
| 8 | ~0.963 | strong |
| 16 | ~0.959 | strong |
| 32 | ~0.893 | plateau / drop |
| 48 | ~0.897 | plateau |
| 62 | ~0.892 | plateau (int64 max) |

**Takeaway (preliminary):** On this net, **small registers (8–16) beat wide ones**.  
Extra bits are not free wins under fixed train defaults + proportional Δv. “16 is special”
is partly “we tried 16,” but **8 is competitive or better** here; 32–62 look saturated
around ~0.89–0.90. Not multi-seed; representation signal only.

## Unary curve (seed 42, ln=none)

| S | best test acc |
|--:|--------------:|
| 8 | ~0.860 |
| 16 | ~0.893 |
| 32 | ~0.924 |
| 64 | ~0.940 |
| 128+ | see `results/v0_5_width_unary/` |

**Takeaway (preliminary):** Unary **keeps improving with S** through at least 64–128
(unlike the register family under the same protocol spirit). Population size and register
width are **not interchangeable** axes.

## Next (Plan §7)

1. Finish unary grid as far as compute allows (256, optionally 512/1024).  
2. `v0_6_encoding` — fixed total bit budget, exp/mantissa splits.  
3. Sparse scaling check only after curves exist.

## Re-run

```bash
python experiments/v0_5_width_register/train.py --ln-mode none --seed 42
python experiments/v0_5_width_unary/train.py --ln-mode none --seed 42
# Summary merges all on-disk per-width JSONs for that ln/seed.
```
