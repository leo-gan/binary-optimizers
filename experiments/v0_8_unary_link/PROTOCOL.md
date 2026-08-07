# Experiment v0.8 — Unary Swarm (link value + XOR writeback)

**Path:** `experiments/v0_8_unary_link/` · **Results:** `results/v0_8_unary_link/`  
**Plan:** `docs/UNARY_SWARM_EXPERIMENT_PLAN.md` · **Terms:** `docs/UNARY_SWARM_TERMINOLOGY.md`  
**WP:** U1 existence proof (defaults also used by v0.9–v0.12)

## Claim

Train MNIST MLP with:

- **Weight** = one bit \(\pm 1\); **swarm** of size \(S\) per **link** (matrix entry).
- **Link value** \(w_{\mathrm{link}} = \mathrm{enc}(s)\), \(s=\sum_k a_k\) (sum only).
- Continuous optimizer **per link** (SGD / SGD+momentum / Adam).
- Writeback: decode \(\Delta\) → flip mask → **XOR** into swarm (equiv. multiply by \(-1\)).

No FP master weight tensor; storage is the int8 swarm only.

## Non-claims

- Not legacy v0.1 majority + pressure-EMA flips.
- Not place-value / binary register Swarm.
- Not structured mant/exp roles inside the swarm.
- Activations remain FP; autograd uses FP for \(w_{\mathrm{link}}\).

## Defaults (WP-U1)

| Item | Value |
|------|-------|
| Data | MNIST |
| Net | Flatten → swarm linear \(784\to128\) → ReLU → swarm linear \(128\to10\) |
| `ln_mode` | `none` |
| \(S\) | 256 |
| Encoder | `fixed`: \(w = s/S\) |
| Optimizer | `sgd`, `lr=0.1` |
| Decoder | `density`: \(p=\mathrm{clamp}(\alpha\|\Delta\|, p_{\min}, p_{\max})+p_{\mathrm{noise}}\) |
| \(\alpha\) | \(10\) |
| \(p_{\min}, p_{\max}\) | \(0\), \(0.25\) |
| \(p_{\mathrm{noise}}\) | \(0.001\) |
| Gain | \(1/\sqrt{\mathrm{fan\_in}}\) |
| Seed | 42 |
| Budget | pure wall (`pure_wall_budget_v1`) |

## XOR convention

Weights stored as int8 \(\{\pm 1\}\).

- **Flip** at position \(k\): \(a_k \leftarrow -a_k\) (equiv. XOR in \(0/1\) packing after map \(\{+1,0\},\{-1,1\}\)).
- Decoder builds a **boolean flip mask** (not a second \(\pm 1\) swarm stored long-term).
- **Directional density (default):** if desired \(\Delta>0\), only **negative** weights may flip (raises \(s\)); if \(\Delta<0\), only **positive** weights may flip (lowers \(s\)). Each eligible site flips with probability \(p\).
- Double flip = identity (tested).

## Grad path

1. Float view of swarm (detached storage) → \(s\) → \(w_{\mathrm{link}}=\mathrm{enc}(s)\).  
2. \(w_{\mathrm{link}}\) retains grad; matmul uses \(w_{\mathrm{link}}\cdot\mathrm{gain}\).  
3. Optimizer reads \(\partial L/\partial w_{\mathrm{link}}\) per link; produces \(\Delta\); never stores FP \(W\).

## Encoder tags

| Tag | Formula |
|-----|---------|
| `fixed` | \(s/S\) |
| `tanh` | \(\tanh(s/\tau)\) default \(\tau=S/2\) |
| `signed_sqrt` | \(\mathrm{sign}(s)\sqrt{|s|/S}\) |
| `majority` | \(\mathrm{sign}(s)\) (ties → \(+1\)); baseline |

## Decoder tags

| Tag | Rule |
|-----|------|
| `density` | \(p=\mathrm{clamp}(\alpha\|\Delta\|,p_{\min},p_{\max})+p_{\mathrm{noise}}\); directional flips |
| `thresholded` | if \(\|\Delta\|>\theta\) then \(p=p_{\max}\) else \(p=p_{\mathrm{noise}}\); directional |
| `sign_noise` | \(p=p_{\mathrm{noise}}\) only; directional via \(\mathrm{sign}(\Delta)\) |

## Optimizer tags

| Tag | State per link |
|-----|----------------|
| `sgd` | none beyond \(g\) |
| `sgd_m` | velocity \(m\) |
| `adam` | \(m, v\) |

## Success (WP-U1)

- Test acc rises above chance; swarm stays in \(\{\pm 1\}^S\).
- Logs: train/test acc, flip fraction, mean \(|\Delta|\).
- Unit tests without dataset: encoder, XOR invertibility, invariants.

## Naming

Human **v0.8** · path **`v0_8_unary_link`**. Follow-ons: `v0_9_unary_decoder`, `v0_10_unary_width`, `v0_11_unary_encoder`, `v0_12_unary_cifar`.
