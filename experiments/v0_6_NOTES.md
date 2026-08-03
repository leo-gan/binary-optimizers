# v0.6 encoding atlas — notes

**Branch:** `feat/v0_6-encoding`  
**Question B:** How to structure a fixed discrete bit budget?

| Item | Choice (from WP1) |
|------|-------------------|
| Scaffold | Register / carry-safe lineage |
| Primary \(n\) | 8, 16 |
| Rescue \(n\) | 32 (optional; not in this run) |
| Unary baseline | \(S=256\) once |
| Protocol (this run, legacy) | seed 42, ln=none, epoch-only patience=5, min_delta=5e-4 then 0 |
| Protocol (going forward) | **wall+epoch budget**: max_wall_sec=1200, max_epochs=80, patience_frac=0.125 |

## Encodings

- `fixed` — pure place-value (v0.3)
- `exp_mant:ne` — per-weight exponent + mantissa, \(w\) scaled into \([-1,1]\)
- `block_scale:ne` — full mantissa budget + shared row exponent
- `unary:S` — v0.1 majority baseline

## Results (primary grid, seed 42)

| tag | best test | epochs | wall (s) |
|-----|----------:|-------:|---------:|
| **fixed_n8** | **0.9633** | 30 | 707 |
| unary_S256 | 0.9582 | 12 | 1665 |
| fixed_n16 | 0.9537 | 31 | 249 |
| block_scale3_n8 | 0.9513 | 16 | 119 |
| block_scale3_n16 | 0.9347 | 27 | 235 |
| exp_mant2_n8 | 0.9242 | 10 | 248 |
| exp_mant2_n16 | 0.9098 | 9 | 82 |
| exp_mant3_n8 | 0.9073 | 8 | 129 |
| exp_mant3_n16 | 0.9040 | 14 | 127 |
| exp_mant4_n8 | 0.8969 | 9 | 82 |
| exp_mant4_n16 | 0.8929 | 10 | 95 |

### Takeaways (MNIST MLP sketch, one seed)

1. **Pure fixed-point wins** at both \(n=8\) and \(n=16\); best overall = **fixed_n8** (matches WP1 register peak).  
2. **Unary \(S=256\)** is second (~0.5% behind fixed_n8); strong baseline, much slower wall.  
3. **Block scale** is closer to fixed-point than per-weight exp/mant, but still behind fixed at same \(n\).  
4. **Per-weight exp_mant hurts** under this train recipe: more \(n_e\) → worse; both \(n=8\) and \(n=16\).  
5. **Float-like structure is not necessary** (and not helpful) on this net for the tested designs — hypothesis “fixed-point is enough for MNIST-scale” is supported for these encodings.

### Caveats

- Single seed; patience=5 atlas ranking. Early-stop used **min_delta=5e-4** on this
  run (could stop while `test > best` by &lt; 5e-4). Fixed afterward: default
  **min_delta=0** for future runs; do not re-run this grid for that alone.  
- Exp/mant decode forces \(|w|\le 1\) via scale normalization — different inductive bias than free float.  
- Optimizer uses STE bit-sum pressure on exp (approximate); a different exp update rule might change the ranking.  
- \(n=32\) rescue **not** run yet (`--include-rescue`).

## Re-run

```bash
python experiments/v0_6_encoding/train.py --ln-mode none --seed 42
python experiments/v0_6_encoding/train.py --include-rescue
```
