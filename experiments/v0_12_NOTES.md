# v0.12 Unary CIFAR flat-MLP probe — run notes (WP-U5)

**Branch:** `plan/unary-swarm`  
**Device:** CPU · seed 42 · pure wall max 2400 s  
**Finished:** 2026-08-06  
**Summary:** `results/v0_12_unary_cifar/summary_seed42.json`

## Setup

| Item | Value |
|------|-------|
| Data | CIFAR-10 |
| Net | Flatten 3072 → H=256 → ReLU → 10 (UnaryLinkLinear) |
| encoder / opt / decoder | fixed / **sgd** / density |
| S grid | 64, 256, 512 |

## Curve

| S | best test | best ep | wall (s) | stop |
|--:|----------:|--------:|---------:|------|
| **64** | **0.3813** | 5 | 1316 | patience_wall |
| 256 | 0.3691 | 4 | 2760 | max_wall_sec |
| 512 | 0.3455 | 2 | 2722 | max_wall_sec |

## Takeaways

1. **Learns above chance** on flat CIFAR MLP (~0.38 best). Absolute acc low (expected for flat MLP + sparse discrete updates).
2. **Larger S under pure wall is confounded:** S=256/512 spend most wall on few epochs (~689 s/ep and ~1360 s/ep); ranking **S=64 > 256 > 512** may reflect **more epochs**, not representation.
3. Preferred S from MNIST (flat ~128) **did not get a clean re-test** at matched epoch budget on CIFAR.
4. Flip rate still ~noise floor; same writeback bottleneck as MNIST runs.

## Follow-ups

- Matched-epoch or longer wall for large S.
- adam + density (U2) and/or majority encoder (U4) on CIFAR.
- Stronger net (conv) if the research question is scale of optima, not flat-MLP ceiling.

## Artifacts

- `results/v0_12_unary_cifar/S*_fixed_seed42.json`
- Log: `logs/v0_12_cifar.log`
