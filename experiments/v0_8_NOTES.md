# v0.8 Unary link Swarm — run notes

**Branch:** `plan/unary-swarm`  
**Device:** CPU (no CUDA in this session)

## WP-U1 existence (`existence_S256`)

| Item | Value |
|------|-------|
| Command | `python experiments/v0_8_unary_link/train.py --ln-mode none --seed 42 --run-tag existence_S256` |
| S / enc / opt / dec | 256 / fixed / sgd / density |
| best test acc | **0.8988** @ epoch 7 |
| epochs / wall | 9 / ~1136 s (patience_wall) |
| flip frac | ~0.0005 (low; α\|\Δ\| small with \|Δ\|~1e-5 + p_noise) |
| Artifact | `results/v0_8_unary_link/existence_S256_seed42.json` |

**Takeaway:** existence holds — multi-level link value + directional XOR learns on MNIST
well above chance. Flip rate is noise-floor dominated under default α=10; may want
higher α or lr for denser updates (WP-U2).

Compare (sketch, different update rule): legacy unary v0.1 at S=32 ~0.92; width
atlas unary S=256 ~0.96 — not a head-to-head under identical budgets.
