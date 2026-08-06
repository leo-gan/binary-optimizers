# v0.8 Unary link Swarm — run notes (WP-U1)

**Branch:** `plan/unary-swarm`  
**Device:** CPU (no CUDA)  
**Finished:** 2026-08-05 (existence cell)

Terminology: [`docs/UNARY_SWARM_TERMINOLOGY.md`](../docs/UNARY_SWARM_TERMINOLOGY.md)  
Plan: [`docs/UNARY_SWARM_EXPERIMENT_PLAN.md`](../docs/UNARY_SWARM_EXPERIMENT_PLAN.md)

## Existence cell (`existence_S256`)

| Item | Value |
|------|-------|
| Command | `python experiments/v0_8_unary_link/train.py --ln-mode none --seed 42 --run-tag existence_S256` |
| S / encoder / opt / decoder | 256 / `fixed` / `sgd` / `density` |
| lr / α / p_max / p_noise | 0.1 / 10 / 0.25 / 0.001 |
| **best test acc** | **0.8988** @ epoch 7 |
| epochs / wall | 9 / ~1136 s (`patience_wall`) |
| mean flip frac | ~0.0005 (≈ noise floor) |
| Artifact | `results/v0_8_unary_link/existence_S256_seed42.json` |

### Takeaways

1. **Existence holds:** sum→link value + per-link SGD + directional XOR learns MNIST well above chance (~0.90 test).
2. **Flip rate is noise-dominated:** \(\alpha|\Delta|\) stays tiny (\(|\Delta|\sim10^{-5}\)–\(10^{-6}\)); almost all flips come from `p_noise=0.001`. Learning still occurs (slow random-walk–like exploration biased by direction).
3. Not a head-to-head with legacy `v0_1` / `v0_5_width_unary` (different update rule and budgets).

### Follow-ups (done later on ladder)

- WP-U2 found **adam+density** and **sgd_m+density** beat plain sgd (~0.93–0.94).
- See `experiments/v0_9_NOTES.md` … `v0_12_NOTES.md` for full ladder results (finished 2026-08-06).
