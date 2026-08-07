# v0.9 Unary decoder / optimizer ablations — run notes (WP-U2)

**Branch:** `plan/unary-swarm`  
**Device:** CPU · seed 42 · pure wall (~1200 s / cell)  
**Finished:** 2026-08-05/06  
**Summary:** `results/v0_9_unary_decoder/summary_seed42.json`

Fixed: net MNIST MLP H=128, S=256, encoder `fixed` (\(w=s/S\)), α=10, p_max=0.25.

## Ranking (best test acc)

| Tag | opt | decoder | p_noise | best test | best ep | wall (s) |
|-----|-----|---------|--------:|----------:|--------:|---------:|
| **adam_density_p0p001** | adam | density | 0.001 | **0.9366** | 8 | 1121 |
| **sgd_m_density_p0p001** | sgd_m | density | 0.001 | **0.9335** | 8 | 1131 |
| sgd_density_p0p001 | sgd | density | 0.001 | 0.8988 | 7 | 1120 |
| sgd_sign_noise_p0p001 | sgd | sign_noise | 0.001 | 0.8969 | 9 | 1267 |
| sgd_thresholded_p0p001 | sgd | thresholded | 0.001 | 0.8968 | 9 | 1246 |
| sgd_density_p0p01 | sgd | density | 0.01 | 0.8852 | 1 | 344 |
| sgd_density_p0 | sgd | density | 0.0 | 0.8813 | 11 | 1255 |

## Default recipe (locked for later WPs when re-running)

| Item | Choice |
|------|--------|
| Optimizer | **adam** (lr=1e-3) — or **sgd_m** (lr=0.1) if simpler |
| Decoder | **density** |
| p_noise | **0.001** |
| Avoid | p_noise=0 (stalls updates when \(\|\Delta\|\) tiny); p_noise=0.01 (too noisy, early peak) |

Under this sketch, **decoder family matters less than optimizer** among density / thresholded / sign_noise at p_noise=0.001 (all ~0.897 with sgd). **Adam and momentum** give the real jump (~+3.5 pp over plain sgd).

## Caveats

- Flip fraction remained ~0.0005 on all cells — noise floor still dominates discrete writeback.
- Adam cell used lr=1e-3; sgd cells used lr=0.1 (protocol defaults).
- Single seed; pure wall on CPU.

## Artifacts

- Per-cell JSON under `results/v0_9_unary_decoder/`
- Log: `logs/v0_9_decoder.log`
