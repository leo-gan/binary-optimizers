# Experiment v0.9 — Unary decoder / optimizer ablations (WP-U2)

**Path:** `experiments/v0_9_unary_decoder/` · **Core:** `experiments/v0_8_unary_link/`  
**Plan:** `docs/UNARY_SWARM_EXPERIMENT_PLAN.md` WP-U2

## Claim

With fixed net, \(S{=}256\), encoder `fixed` (\(w=s/S\)), rank:

- optimizers: `sgd` · `sgd_m` · `adam`
- decoders: `density` · `thresholded` · `sign_noise`
- noise floor: `0` · `0.001` · `0.01` (sparse; default grid uses a subset)

One change per cell. Default train recipe from v0.8 otherwise.

## Default sparse grid

| Tag | opt | decoder | p_noise | notes |
|-----|-----|---------|---------|-------|
| `sgd_density` | sgd | density | 0.001 | WP-U1 baseline |
| `sgd_m_density` | sgd_m | density | 0.001 | |
| `adam_density` | adam | density | 0.001 | lr default 1e-3 for adam |
| `sgd_thresholded` | sgd | thresholded | 0.001 | |
| `sgd_sign_noise` | sgd | sign_noise | 0.001 | |
| `sgd_density_p0` | sgd | density | 0 | no noise floor |
| `sgd_density_p01` | sgd | density | 0.01 | medium noise |

## Acceptance

- Ranking of opt×decoder by best test acc (seed 42, pure wall).
- Lock **default recipe** for WP-U3 width atlas (document in NOTES after first run).

## Non-claims

- Not a full \(\alpha,p_{\max}\) grid (optional CLI).
- Not multi-seed.
