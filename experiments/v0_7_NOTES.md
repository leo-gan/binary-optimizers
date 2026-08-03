# v0.7 CIFAR encoding probe — notes

**Branch:** `feat/v0_7-cifar-encoding`  
**ID:** `v0_7_cifar_encoding`  
**Question:** Does fixed-point still beat exp/mant when data is harder (CIFAR-10)?

## Setup

| Item | Value |
|------|--------|
| Net | Flat MLP 3072→128→10 (not a convnet) |
| Data | CIFAR-10, seed 42, `ln_mode=none` |
| Budget | pure wall **2400 s**, stall **300 s** (option B) |
| Cells | fixed@8, fixed@16, exp_mant:2@8, exp_mant:2@16 |

## Results

| tag | best test | epochs (best@) | wall | stop |
|-----|----------:|----------------|-----:|------|
| **fixed_n8** | **0.5088** | 65 (40) | 832 s | patience_wall |
| fixed_n16 | 0.4910 | 162 (152) | 2404 s | max_wall_sec |
| exp_mant2_n8 | 0.4409 | 101 (74) | 1164 s | patience_wall |
| exp_mant2_n16 | 0.4196 | 58 (40) | 979 s | patience_wall |

### vs MNIST (v0.6, same encodings, easier data)

| tag | MNIST | CIFAR | gap |
|-----|------:|------:|----:|
| fixed_n8 | 0.963 | 0.509 | hard task drop |
| fixed_n16 | 0.954 | 0.491 | |
| exp_mant2_n8 | 0.924 | 0.441 | |
| exp_mant2_n16 | 0.910 | 0.420 | |

**Ranking unchanged:** fixed ≫ exp_mant; **n=8 ≥ n=16** for fixed; more exp budget still hurts.

## Takeaway

On this flat CIFAR MLP and train recipe, **exp/mant does not overtake fixed-point**.
WP2’s “fixed-point is enough / better for these designs” **holds under harder data**
at sketch quality—absolute accuracy is low (flat MLP on CIFAR is a weak vision net),
but the **relative** encoding ranking is the research question.

## Caveats

- Flat MLP is a stress toy, not a CIFAR SOTA stack.  
- Single seed; pure wall option B.  
- exp_mant still uses \|w\|≤1 normalization + STE-ish exp updates from v0.6.
