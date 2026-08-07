# v0.11 Unary sum-encoder family — run notes (WP-U4)

**Branch:** `plan/unary-swarm`  
**Device:** CPU · seed 42 · pure wall  
**Finished:** 2026-08-06  
**Summary:** `results/v0_11_unary_encoder/summary_seed42.json`

Fixed: S=256, sgd lr=0.1, density, p_noise=0.001 (same pre-adam recipe as U3).

## Ranking

| Encoder | Formula | best test | best ep | wall (s) |
|---------|---------|----------:|--------:|---------:|
| **majority** | \(\mathrm{sign}(s)\), ties → +1 | **0.9377** | 5 | 832 |
| signed_sqrt | \(\mathrm{sign}(s)\sqrt{\|s\|/S}\) | 0.9122 | 7 | 1055 |
| tanh | \(\tanh(s/\tau)\), \(\tau=S/2\) | 0.9115 | 7 | 1051 |
| fixed | \(s/S\) | 0.8988 | 7 | 1038 |

## Takeaways

1. With **same XOR writeback**, **majority** link values beat multi-level sum maps on this MNIST sketch (~+4 pp over fixed).
2. Compressive maps (tanh, signed_sqrt) sit between majority and pure fixed.
3. Multi-level fixed is **not** free lunch here: continuous optimizer + sparse flips may prefer coarse binary link values.
4. Encoder input remains **sum only** (no bitfield roles) — still holds.

## Caveats

- Recipe was **sgd**, not U2 **adam**. Majority under adam is an open check.
- Single seed.

## Artifacts

- `results/v0_11_unary_encoder/enc_*_seed42.json`
- Log: `logs/v0_11_encoder.log`
