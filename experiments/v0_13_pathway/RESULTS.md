# v0.13 results

All numbers MNIST, \(H=128\), `PathwayOptimizer` (\(k=8\), \(\lambda=0.1\),
`ste_topk`), seed noted. Wall protocol `pure_wall_budget_v1`. Artifacts
under `results/v0_13_pathway/` (gitignored).

This file is the scoreboard. Isolation **pass rule** is pre-registered
below; seeds `{0,1}` were not run when this section was written.

---

## 1. Flip-search existence (`--norm none`)

| Tag | Seed | Best test | Best ep | Stop | Wall |
|-----|-----:|----------:|--------:|------|-----:|
| `smoke` | 42 | 0.5158 | 1 | max_wall 60 s | 77 s |
| `pathway` | 42 | **0.6103** | 11 | patience_wall 150 s | 1035 s |

Chance is 0.10. Latent-free ±1 flip search learns. \(\widehat{A}(W)\) stayed
1.0017 (zlib floor). Accept ~0.28–0.32. Train CE ~44 (unscaled ±1 logits).
Not a Swarm comparison (v0.1 LN0 is 0.9239).

---

## 2. PathwayNorm, short patience (do not use)

| Tag | Seed | Best test | Stop |
|-----|-----:|----------:|------|
| `pathway_norm` | 42 | 0.5573 @ ep 1 | patience_wall after 4 ep / 230 s |

Test never beat epoch 1. Train CE ~3 (scale works). **Invalid as a norm
comparison** — the 150 s stall cut the run.

---

## 3. Two-way bake-off vs LayerNorm (contaminated PathwayNorm)

`compare_norms.py`, seed 42, 1200 s, `patience_frac=1`.

| Tag | Norm | Best test | Best ep | Final CE |
|-----|------|----------:|--------:|---------:|
| `cmp_pathway` | PathwayNorm, **EMA on every trial flip** | 0.6368 | 14 | 2.220 |
| `cmp_layernorm` | LayerNorm, no affine | **0.6514** | 16 | 1.727 |

LN +1.46 pp. **Superseded for PathwayNorm:** rejected flips wrote the motif
EMA. LayerNorm cell is still the standard-norm reference for seed 42.

---

## 4. Tracking fix and motif ablation (seed 42)

Same wall as §3. Train loop: `track=False` during closures; `update_stats`
once on the accepted hidden state.

| Tag | Arm | Best test | Best ep | Accept | vs LN |
|-----|-----|----------:|--------:|-------:|------:|
| `cmp_pathway_fix` | PathwayNorm, motif on, accepted-state EMA | **0.6750** | 16 | ~0.40 | +2.36 pp |
| `cmp_pathway_l1` | L1 scale only (`strength=0`) | 0.6732 | 13 | ~0.32 | +2.18 pp |
| `cmp_layernorm` | LayerNorm (§3) | 0.6514 | 16 | ~0.33 | — |

**Read this as:** a hidden norm helps; **motif vs L1 is a tie** (0.18 pp).
The +2.4 pp vs LN on this seed is **not** evidence that the pathway/motif
story beats LayerNorm as a general norm.

---

## 5. Isolation panel (can the motif story pass?)

**Question:** does the **motif subtract** beat both (a) the same L1 scale
without motif and (b) LayerNorm, on more than one seed?

**Arms** (identical optimizer, wall, data):

| Arm id | How |
|--------|-----|
| `layernorm` | `--norm layernorm` |
| `scale_only` | `--norm pathway --norm-strength 0` |
| `pathway` | `--norm pathway --norm-strength 1` (accepted-state EMA) |

**Budget:** 1200 s wall, `patience_frac=1`.  
**Seeds:** `{42, 0, 1}`. Seed 42 **reuses** §3–§4 JSONs (no rerun).  
**Metric:** best test accuracy of the run.

**Pass rule (fixed before seeds 0 and 1):**

1. \(\overline{\mathrm{acc}}(\texttt{pathway})-\overline{\mathrm{acc}}(\texttt{scale\_only})\ge 0.02\)
2. \(\overline{\mathrm{acc}}(\texttt{pathway})-\overline{\mathrm{acc}}(\texttt{layernorm})\ge 0.02\)
3. \(\mathrm{acc}(\texttt{pathway})>\mathrm{acc}(\texttt{scale\_only})\) on at least 2 of 3 seeds

Fail ⇒ this motif is not why we beat LN. Do **not** retune \(k\)/\(\lambda\)
to salvage the claim.

```bash
python experiments/v0_13_pathway/isolate_motif.py --seeds 42,0,1
```

Writes `results/v0_13_pathway/isolate_motif_summary.{json,md}`.

### Verdict: **FAIL**

| seed | layernorm | scale_only | pathway | pathway − scale | pathway − LN |
|-----:|----------:|-----------:|--------:|----------------:|-------------:|
| 42 | 0.6514 | 0.6732 | 0.6750 | +0.18 pp | +2.36 pp |
| 0 | 0.6667 | 0.6564 | 0.6714 | +1.50 pp | +0.47 pp |
| 1 | **0.6865** | 0.6697 | 0.6540 | −1.57 pp | −3.25 pp |
| **mean** | **0.6682** | 0.6664 | 0.6668 | **+0.04 pp** | **−0.14 pp** |

Motif > L1 on 2/3 seeds, but both mean deltas are ≪ 2 pp. Mean PathwayNorm
≈ mean L1 ≈ mean LayerNorm. The motif subtract is **not** why anyone
beats LN; on seed 1 LayerNorm is the best arm.

Tags: `iso_*_s{0,1}` plus reused `cmp_layernorm_seed42`,
`cmp_pathway_l1_seed42`, `cmp_pathway_fix_seed42`.
