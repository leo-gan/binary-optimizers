# v0.6 encoding atlas — notes

**Branch:** `feat/v0_6-encoding`  
**Question B:** How to structure a fixed discrete bit budget?

| Item | Choice (from WP1) |
|------|-------------------|
| Scaffold | Register / carry-safe lineage |
| Primary \(n\) | 8, 16 |
| Rescue \(n\) | 32 (optional) |
| Unary baseline | \(S=256\) once |

## Encodings implemented

- `fixed` — pure place-value (v0.3)
- `exp_mant:ne` — per-weight exponent + mantissa, \(w\) scaled into \([-1,1]\)
- `block_scale:ne` — full mantissa budget + shared row exponent
- `unary:S` — v0.1 majority baseline

## Results

*(fill after train runs)*

```bash
python experiments/v0_6_encoding/train.py --ln-mode none --seed 42
python experiments/v0_6_encoding/train.py --include-rescue
```
