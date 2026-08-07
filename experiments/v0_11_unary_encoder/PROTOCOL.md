# Experiment v0.11 — Unary sum-encoder family (WP-U4)

**Path:** `experiments/v0_11_unary_encoder/` · **Core:** `v0_8_unary_link`  
**Plan:** WP-U4

## Claim

Encoder input remains **scalar sum** \(s\). Compare 1D maps at fixed \(S{=}256\),
density decoder, SGD (or recipe from v0.9):

| Tag | \(\mathrm{enc}(s)\) |
|-----|---------------------|
| `fixed` | \(s/S\) |
| `tanh` | \(\tanh(s/\tau)\), \(\tau=S/2\) |
| `signed_sqrt` | \(\mathrm{sign}(s)\sqrt{|s|/S}\) |
| `majority` | \(\mathrm{sign}(s)\) (ties → +1); baseline under same XOR writeback |

## Non-claims

- Not bitfield exp/mant roles inside the swarm.
