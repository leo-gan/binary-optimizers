# v0.5 width atlas — notes

**Branch:** `feat/v0_5-width-atlas`  
**Question A:** How much discrete state per weight? Two families, not mixed.

| Experiment | Coding | Width axis |
|------------|--------|------------|
| `v0_5_width_register` | v0.3 carry-safe place-value | \(n_{\mathrm{bits}}\) |
| `v0_5_width_unary` | v0.1 unary majority Swarm | population \(S\) |

Protocol: MNIST MLP `hidden=128`, `ln_mode=none`, seed 42, early stop **80 / patience=5**.  
**No per-width hyperparam search.** Register runs scale `max_step` / `step_scale` with
\(v_{\max}\) so \(\Delta v / v_{\max}\) matches the n=16 reference (fair width compare).

**Why patience=5 (not 10):** width atlas ranks representations, not max-polished accuracy.
Retrofit on prior histories: ranking unchanged; a few absolute scores drop ≤~1.2%; wall
time roughly halves on late-improving widths. Use `--patience 10` only when polishing a
chosen width.

## Register curve (seed 42, ln=none)

| n_bits | best test acc | epochs | note |
|-------:|--------------:|-------:|------|
| 1 | 0.6790 | 8 | p=5 |
| 2 | 0.8804 | 12 | p=5 |
| 3 | 0.9134 | 7 | p=5 |
| 4 | 0.9450 | 12 | p=5 |
| 6 | 0.9594 | 14 | p=5 |
| 8 | **0.9633** | 35 | p=10 (legacy) |
| 16 | 0.9589 | 74 | p=10 |
| 32 | 0.8926 | 29 | p=10 |
| 48 | 0.8972 | 52 | p=10 |
| 62 | 0.8923 | 40 | p=10 |

**Takeaway:** Clear **unimodal** shape: rises 1→8, peaks at **n≈8**, slight drop at 16,
then **cliff** to ~0.89 for n≥32. Left side (1–6) was the missing half; optimum is
**not** “smaller always better.” Wide registers still lose under proportional Δv.
Caveat: peak n=8 used patience 10; n=6 used patience 5 — gap is small (~0.4%).
Not multi-seed.

## Unary curve (seed 42, ln=none)

| S | best test acc | epochs | note |
|--:|--------------:|-------:|------|
| 8 | 0.8598 | 13 | p=10 |
| 16 | 0.8930 | 11 | p=10 |
| 32 | 0.9239 | 29 | p=10 |
| 64 | 0.9398 | 16 | p=10 |
| 128 | 0.9570 | 35 | p=10 |
| 256 | 0.9617 | 26 | p=10 |
| 512 | 0.9592 | 9 | p=5 |
| 1024 | **0.9638** | 21 | p=5 |

**Takeaway:** Strong gains **S=8→128**, then **soft plateau ≈256–1024**
(+0.0047 from 128→256; −0.0025 at 512; +0.0021 256→1024 — noise-scale at one seed).
**Practical optimum band: S ≈ 256–1024** (diminishing returns; 1024 is ~2× wall of 512
per epoch). No need for S=2048 under this protocol. Population size and register width
remain **non-interchangeable** (register peaks near n=8; unary wants large S).

## Next (Plan §7 — see `docs/PLAN.md`)

1. **`v0_6_encoding` (WP2):** register scaffold; lock \(n\in\{8,16\}\) (+ optional 32 rescue);
   vary fixed-point vs exp/mant vs block scale; unary \(S=256\) baseline only.  
2. Optional: densify register \(n\in\{7,9,10,12\}\) with matched patience.  
3. Sparse WP3 scaling check only after encoding curves exist.

## Re-run

```bash
python experiments/v0_5_width_register/train.py --ln-mode none --seed 42
python experiments/v0_5_width_unary/train.py --ln-mode none --seed 42
# Summary merges all on-disk per-width JSONs for that ln/seed.
```
