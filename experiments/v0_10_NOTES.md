# v0.10 Unary swarm-size atlas — run notes (WP-U3)

**Branch:** `plan/unary-swarm`  
**Device:** CPU · seed 42 · pure wall  
**Finished:** 2026-08-06  
**Summary:** `results/v0_10_unary_width/summary_seed42.json`

## Recipe used (not yet U2 adam lock)

| Item | Value |
|------|-------|
| encoder | `fixed` |
| opt / lr | **sgd** / 0.1 |
| decoder | density, α=10, p_max=0.25, p_noise=0.001 |
| ln_mode | none |

U3 ran **before** adopting adam as default; curve is for **sgd+density** only.

## Acc vs S

| S | best test | best ep | wall (s) |
|--:|----------:|--------:|---------:|
| 8 | 0.8978 | 7 | 211 |
| 16 | 0.8993 | 12 | 296 |
| 32 | 0.8999 | 9 | 328 |
| 64 | 0.8985 | 9 | 459 |
| **128** | **0.9000** | 10 | 790 |
| 256 | 0.8988 | 7 | 1025 |
| 512 | 0.8964 | 6 | 1423 |
| 1024 | 0.8910 | 3 | 1295 |

**Peak sample:** S=128 at **0.9000**.

## Takeaways

1. Under sgd+density XOR path, accuracy is **nearly flat** (~0.89–0.90) from S=8 to 512.
2. **No large gain from large S** (contrast legacy unary majority+flip atlas, which rose strongly to S≈256–1024).
3. S=1024 slightly worse and **epoch-starved** under pure wall (3 epochs only).
4. Practical pick for this recipe: **S≈32–128** (cheap, same acc band).

## Do not merge with

`results/v0_5_width_unary` / legacy v0.1 — different update rule and often different budgets.

## Follow-up

Re-run width atlas with **adam + density** (U2 winner) to see if preferred S moves.

## Artifacts

- `results/v0_10_unary_width/S*_seed42.json`
- Log: `logs/v0_10_width.log`
