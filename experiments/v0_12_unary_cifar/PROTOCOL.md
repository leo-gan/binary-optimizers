# Experiment v0.12 — Unary link Swarm CIFAR scale probe (WP-U5)

**Path:** `experiments/v0_12_unary_cifar/` · **Core:** `v0_8_unary_link`  
**Plan:** WP-U5

## Claim

Sparse scale check: does preferred \(S\) / encoder ranking move on harder data?

| Item | Default |
|------|---------|
| Data | CIFAR-10 (flat MLP on 3072-d) |
| Net | Flatten → swarm \(3072\to H\) → ReLU → swarm \(H\to10\) |
| \(H\) | 256 |
| \(S\) grid (sparse) | 64, 256, 512 |
| encoder | `fixed` (optional majority cell) |
| opt / decoder | sgd + density (WP-U1 recipe) |
| Budget | pure wall; default max_wall_sec=2400 (CIFAR option B) |
| seed | 42 |

## Acceptance

- Relative ranking note: does larger \(S\) still help vs MNIST sketch?
- Absolute acc expected low on flat MLP; ranking > polish.

## Non-claims

- Not a strong conv CIFAR model (future work).
- Not multi-seed.
