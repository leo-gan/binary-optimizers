# Experiment v0.10 — Unary swarm size atlas (WP-U3)

**Path:** `experiments/v0_10_unary_width/` · **Core:** `v0_8_unary_link`  
**Plan:** WP-U3

## Claim

Vary **only** swarm size \(S\) under the locked Unary link recipe:

\[
S \in \{8, 16, 32, 64, 128, 256, 512, 1024\}
\]

Default recipe (until v0.9 ranking overrides):

| Item | Value |
|------|-------|
| encoder | `fixed` |
| opt | `sgd` |
| lr | 0.1 |
| decoder | `density` |
| \(\alpha, p_{\max}, p_{\mathrm{noise}}\) | 10, 0.25, 0.001 |
| ln_mode | none |
| seed | 42 |

Do **not** merge tables with legacy `v0_5_width_unary` without labeling (different update rule).

## Acceptance

- acc vs \(S\) curve + plateau / peak statement for **this** model.
