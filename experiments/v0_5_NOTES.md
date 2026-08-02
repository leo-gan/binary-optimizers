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

| n_bits | best test acc | epochs |
|-------:|--------------:|-------:|
| 8 | 0.9633 | 35 |
| 16 | 0.9589 | 74 |
| 32 | 0.8926 | 29 |
| 48 | 0.8972 | 52 |
| 62 | 0.8923 | 40 |

**Takeaway:** On this net, **small registers (8–16) beat wide ones**. Extra bits are not
free wins under fixed train defaults + proportional Δv. “16 is special” is partly “we
tried 16,” but **8 is competitive or better** here; 32–62 plateau around ~0.89–0.90.
Not multi-seed; representation signal only.

## Unary curve (seed 42, ln=none)

| S | best test acc | epochs |
|--:|--------------:|-------:|
| 8 | 0.8598 | 13 |
| 16 | 0.8930 | 11 |
| 32 | 0.9239 | 29 |
| 64 | 0.9398 | 16 |
| 128 | 0.9570 | 35 |
| 256 | 0.9617 | 26 |

**Takeaway:** Unary **keeps improving with S** through 256 (gains slow after 128).
Unlike the register family, more agents help. Population size and register width are
**not interchangeable** axes. Optional 512/1024 left for later if wanted.

## Next (Plan §7)

1. Optional: unary S=512/1024 if compute allows.  
2. `v0_6_encoding` — fixed total bit budget, exp/mantissa splits.  
3. Sparse scaling check only after encoding curves exist.

## Re-run

```bash
python experiments/v0_5_width_register/train.py --ln-mode none --seed 42
python experiments/v0_5_width_unary/train.py --ln-mode none --seed 42
# Summary merges all on-disk per-width JSONs for that ln/seed.
```
