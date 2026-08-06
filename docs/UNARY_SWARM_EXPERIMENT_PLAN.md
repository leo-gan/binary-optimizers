# Unary Swarm — experiment plan

**Branch:** `plan/unary-swarm`  
**Status:** plan (not implemented)  
**Date:** 2026-08-05  
**Terminology (frozen):** [UNARY_SWARM_TERMINOLOGY.md](UNARY_SWARM_TERMINOLOGY.md)  
**Related:** [PLAN.md](PLAN.md) (repo roadmap), [MEMORY_1B.md](MEMORY_1B.md), [TRAIN_BUDGET.md](TRAIN_BUDGET.md)

This plan defines the **first experiment ladder** for the **intended** Unary Swarm
model (sum encoder → link value; Adam/SGD per link; decoder + **XOR** writeback).
It is **not** a re-run of `experiments/v0_1` (majority ±1 + pressure-EMA flips).

---

## 1. Goal

### 1.1 Research claim

Train a net where:

1. **Stored state** per matrix entry (**link**) is only a **swarm** of \(S\) **weights**
   (bits \(\pm 1\)).
2. **Link value** used in the matmul is \(w_{\mathrm{link}} = \mathrm{enc}(s)\) with
   \(s = \sum_k a_k\) (**sum/popcount only**).
3. A **continuous optimizer** (SGD or Adam) runs **per link** on \(w_{\mathrm{link}}\)
   (state \(g\), and \(m\) / \(v\) if used).
4. The continuous update \(\Delta\) is turned into an **update swarm** and merged with
   **XOR** (plus noise so small steps still rarely flip bits).

**Success:** the discrete swarm tracks a useful multi-level link value and learns on
MNIST at least as stably as legacy unary v0.1 at matched \(S\), with a clear story for
how \(\|\Delta\|\) controls flip density.

### 1.2 Non-goals (this plan)

| Out of scope now | Why |
|------------------|-----|
| Place-value / binary register Swarm | Separate thesis (fixed-point-in-bits) |
| Structured mant/exp **roles inside** the swarm | Encoder input locked to **sum only** |
| Bit packing for production memory claims | Engineering after learning works |
| Multi-seed error bars / STE leaderboards | Representation existence first |
| CIFAR / conv scale | After MNIST existence (phase B) |
| Pure integer backward | Longer-term grand goal |

---

## 2. Design freeze (do not re-argue in PROTOCOLs)

From [UNARY_SWARM_TERMINOLOGY.md](UNARY_SWARM_TERMINOLOGY.md):

| Item | Lock |
|------|------|
| Weight | One bit \(\pm 1\) |
| Swarm | Batch of \(S\) weights on one **link** |
| Link | Matrix entry; owns one swarm + optional \(g,m,v\) |
| Link value | \(w_{\mathrm{link}} = \mathrm{enc}(s)\), \(s=\sum a_k\) |
| Encoder input | **Sum/popcount only** |
| Optimizer attachment | **Per link** |
| Merge \(\star\) | **XOR** (until a later experiment changes it) |

### 2.1 Intended step (reference)

```text
swarm  --enc(s)-->  w_link  --forward/backward-->  g
per-link Adam/SGD  -->  Δ
Δ (+ noise)  --decoder-->  update swarm u ∈ {±1}^S
swarm  ←  swarm XOR u
```

### 2.2 vs legacy `v0_1`

| Stage | This plan | `v0_1` |
|-------|-----------|--------|
| \(w_{\mathrm{link}}\) | Multi-level \(\mathrm{enc}(s)\) | \(\mathrm{sign}(s)\) |
| Optimizer | SGD / Adam on link | Flip + pressure EMA |
| Writeback | Decoder + XOR | Probabilistic sign flips |

New code lives under a **new experiment id** (proposed: `v0_8_unary_link`), not under
`v0_1/`.

---

## 3. Hypotheses

| ID | Hypothesis | How to falsify |
|----|------------|----------------|
| H1 | Sum-encoded multi-level \(w_{\mathrm{link}}\) + XOR writeback can train MNIST MLP above chance and above a frozen-random swarm baseline | Best test acc stays ~chance or &lt; baseline |
| H2 | Fixed encoder \(w_{\mathrm{link}}=s/S\) is enough for a first existence proof | No learning until a different \(f(s)\) is used |
| H3 | Flip density should track \(\|\Delta\|\) (larger steps → more XOR 1-bits) | Measured flip rate independent of \(\|\Delta\|\) |
| H4 | Adam per link is more stable than SGD for the same decoder | SGD fails or needs much more tuning; Adam trains |
| H5 | Larger \(S\) improves fidelity of \(\Delta\)→swarm (smoother effective steps) until a plateau | Acc vs \(S\) flat or inverse for all reasonable decoders |

---

## 4. Work packages

### WP-U0 — Spec + scaffolding  ← **done**

| Deliverable | Status |
|-------------|--------|
| Frozen terminology | **Done** — `docs/UNARY_SWARM_TERMINOLOGY.md` |
| Experiment plan | **Done** — this file |
| Memory context (1B static) | **Done** — `docs/MEMORY_1B.md` |
| Experiment folder skeleton | **Done** — `experiments/v0_8_unary_link/` |

### WP-U1 — Existence proof (MNIST, single cell)  ← **implemented + run**

**Code:** `experiments/v0_8_unary_link/` (`runner.py`, `train.py`, unit tests).  
**Notes:** `experiments/v0_8_NOTES.md`. **Result:** best test **0.8988** (S=256 fixed sgd density).

**Goal:** one configuration that trains and is not a bug.

| Item | Default (proposed) |
|------|--------------------|
| Data | MNIST |
| Net | Flatten → Linear-link swarm \(784\to 128\) → ReLU → Linear-link swarm \(128\to 10\) |
| `ln_mode` | `none` |
| \(S\) | **256** (plateau pick from legacy unary atlas; re-check under new update) |
| Encoder | \(w_{\mathrm{link}} = s/S\) (fixed) |
| Optimizer | **SGD** first (fewer moving parts); optional Adam cell |
| Gain | \(1/\sqrt{\mathrm{fan\_in}}\) on link value (match v0.1 stability) |
| Seed | 42 |
| Budget | Prefer **pure wall** (`docs/TRAIN_BUDGET.md`); else epoch patience for first debug |

**Decoder v0 (must implement something concrete):**

Minimal proposal to unblock code (can be replaced later without renaming terms):

1. Map \(\Delta\) to a target flip count or probability  
   \(p = \mathrm{clamp}(\alpha |\Delta|,\, p_{\min},\, p_{\max})\).  
2. Sample update swarm \(u\): each bit \(+1\) (“apply XOR flip”) with prob \(p\), else
   identity bit for XOR (see packing note below).  
3. \(\mathrm{swarm} \leftarrow \mathrm{swarm}\ \mathrm{XOR}\ u\).  
4. Optional: independent noise floor \(p_{\mathrm{noise}}\ll p_{\max}\) so tiny \(\Delta\) still
   rarely flips.

**XOR on \(\pm 1\):** define bit-level XOR in \(0/1\) packing, or equivalently
“multiply by \(-1\) where update says flip.” PROTOCOL must state one convention.

**Acceptance (WP-U1):**

- Test acc rises and plateaus (not pure noise).  
- After every step: swarm still in \(\{\pm 1\}^S\).  
- Log mean \(|\Delta|\), mean flip fraction, train/test acc.  
- Unit tests: encoder sum, XOR invertibility (double XOR = identity), swarm invariant.

### WP-U2 — Decoder and optimizer ablations  ← **implemented + run**

**Code:** `experiments/v0_9_unary_decoder/`. **Notes:** `experiments/v0_9_NOTES.md`.

Fix net + \(S{=}256\) + encoder \(s/S\). Vary **only** one axis per grid.

| Axis | Cells (sparse) |
|------|----------------|
| Optimizer | SGD · SGD+momentum · Adam |
| Decoder | density \(\propto|\Delta|\) · thresholded · sign(\(\Delta\))+noise only |
| Noise floor | 0 · small · medium |
| \(\alpha, p_{\max}\) | small bracket around WP-U1 working point |

**Acceptance:** ranking of optimizer/decoder classes + one **default recipe** for later width work.

**Result (seed 42, CPU):** **adam+density 0.9366** > sgd_m+density 0.9335 > sgd+density 0.8988.  
**Locked recipe:** adam (lr=1e-3) + density + p_noise=0.001.

### WP-U3 — Swarm size atlas (Question A for *this* model)  ← **implemented + run (sgd)**

**Code:** `experiments/v0_10_unary_width/`. **Notes:** `experiments/v0_10_NOTES.md`.

Vary \(S\) only under the WP-U2 default recipe:

\[
S \in \{8, 16, 32, 64, 128, 256, 512, 1024\}
\]

(Same spirit as `v0_5_width_unary`, but **new** update rule — do not merge result
tables with v0.1 without a clear label.)

**Acceptance:** acc vs \(S\) curve + statement of plateau / peak for **this** Unary Swarm.

**Result (seed 42, sgd+density — pre-adam lock):** nearly flat **~0.89–0.90**; peak sample
**S=128 @ 0.9000**. Large S does not help; S=1024 wall-starved.

### WP-U4 — Encoder family (still sum-only)  ← **implemented + run (sgd)**

**Code:** `experiments/v0_11_unary_encoder/`. **Notes:** `experiments/v0_11_NOTES.md`.

Encoder input remains **scalar \(s\)**. Compare 1D maps:

| Tag | \(\mathrm{enc}(s)\) |
|-----|---------------------|
| `fixed` | \(s/S\) |
| `tanh` | \(\tanh(s/\tau)\) (one \(\tau\)) |
| `signed_sqrt` | \(\mathrm{sign}(s)\sqrt{|s|/S}\) (example compressive map) |

**Not** bitfield exp/mant. Optional: compare to legacy majority \(w=\mathrm{sign}(s)\) as a
**baseline encoder** under the **same** XOR writeback (isolates encode vs update).

**Result (seed 42, S=256, sgd):** **majority 0.9377** > signed_sqrt 0.9122 > tanh 0.9115 >
fixed 0.8988.

### WP-U5 — Scale probe  ← **implemented + run (budget-confounded)**

**Code:** `experiments/v0_12_unary_cifar/`. **Notes:** `experiments/v0_12_NOTES.md`.

- Harder data under pure wall (default 2400 s).  
- Ask: does preferred \(S\) move?  
- Sparse, one seed, ranking not polish.

**Result (seed 42, sgd+fixed):** S=64 **0.3813** > S=256 0.3691 > S=512 0.3455.
Large-S cells hit max wall with few epochs — ranking confounded by budget.

---

## 5. Proposed experiment IDs

| ID | Focus | Depends on |
|----|--------|------------|
| `v0_8_unary_link` | WP-U1 existence + PROTOCOL | terminology freeze |
| `v0_8_1_unary_link` | Pure-wall re-run of winner (if epoch budget used first) | `v0_8` |
| `v0_9_unary_decoder` | WP-U2 ablations | `v0_8` |
| `v0_10_unary_width` | WP-U3 \(S\) atlas | default recipe from U2 |
| `v0_11_unary_encoder` | WP-U4 1D \(\mathrm{enc}(s)\) | default recipe |
| `v0_12_unary_cifar` | WP-U5 CIFAR flat MLP probe | U1–U3 recipe |

Paths: `experiments/<id>/` with `PROTOCOL.md`, `README.md`, `train.py`, `test_*.py`,
results under `results/<id>/` (gitignored). Shared loop: `v0_8_unary_link/runner.py`.

---

## 6. Protocol spirit

| Rule | Detail |
|------|--------|
| One change per cell | Do not co-vary \(S\), decoder, and lr |
| No per-cell lr search | Fix train defaults; only break if representation is broken |
| Pure wall preferred | [TRAIN_BUDGET.md](TRAIN_BUDGET.md) for published rankings |
| Seed | 42 for sketches |
| Logging | best test acc, curve, flip fraction, mean \(\|\Delta\|\), wall time |
| Store | DuckDB optional via `binary_optimizers.store` when useful |

### 6.1 Suggested metrics

| Metric | Purpose |
|--------|---------|
| Best / final test accuracy | Primary |
| Train accuracy / loss | Underfit vs diverge |
| Mean flip fraction per step | Decoder ↔ \(\Delta\) (H3) |
| Corr(\(\|\Delta\|\), flip frac) | H3 quantitative |
| Mean \(\|s\|/S\) or histogram of \(w_{\mathrm{link}}\) | Are link values stuck at ±1? |
| Wall seconds | Fair budget |

### 6.2 Baselines (same net / data / budget)

| Baseline | Role |
|----------|------|
| Legacy `v0_1` at same \(S\) | Old unary update |
| STE + SGD on FP master (matched width story optional) | Continuous upper reference |
| Frozen random swarm (no XOR updates) | Sanity lower bound |
| Majority encoder + XOR writeback | Encoder-only ablation |

---

## 7. Implementation sketch (for implementers)

Suggested module split under `experiments/v0_8_unary_link/`:

| File | Responsibility |
|------|----------------|
| `layers.py` | Swarm buffer `[out,in,S]`; `link_value()` from sum; forward with STE/path for \(g\) on \(w_{\mathrm{link}}\) |
| `optimizer.py` | Per-link SGD/Adam on link-value grads; build \(\Delta\); decode; XOR into swarm |
| `model.py` | MNIST MLP topology (match v0.1 shape) |
| `train.py` | CLI, pure wall hooks, logging JSON |
| `PROTOCOL.md` | Normative; wins over this plan on conflict after implementation |
| `test_v0_8_unary_link.py` | Invariants, no dataset required in CI |

**Grad path (decision at implement time, document in PROTOCOL):**

- Preferred for clarity: treat \(w_{\mathrm{link}}\) as the differentiable quantity (recompute
  from swarm each forward); STE only if bits need direct grads.  
- Optimizer steps in \(w_{\mathrm{link}}\)-space conceptually; **storage** remains the swarm
  after XOR (no FP master tensor retained as the weight).

**Adam/SGD state:** allocate \(m\) (and \(v\)) with shape `[out, in]` per swarm layer —
**per link**, not `[out, in, S]`.

---

## 8. Memory note (plan-level)

Under this design, static weight-path state per link is roughly:

| Buffer | Scale |
|--------|-------|
| Swarm | \(S\) bits (packed ideal) or \(S\) bytes (int8 research) |
| \(g\) | 1 float / link |
| \(m\) (SGD-M or Adam) | 1 float / link |
| \(v\) (Adam) | 1 float / link |

So unlike pure flip+EMA unary, **Adam reintroduces FP moments per link**. Update
[MEMORY_1B.md](MEMORY_1B.md) with an explicit “Unary link-value + Adam” row when WP-U1
lands (do not pretend moments are free).

---

## 9. Immediate next actions (ordered)

1. **Land this plan + terminology + related docs on `plan/unary-swarm`.**  
2. Open `experiments/v0_8_unary_link/` with PROTOCOL.md stating encoder, XOR convention,
   decoder v0 formulas, and train defaults.  
3. Implement layers + optimizer + tests (no dataset in unit tests).  
4. Run WP-U1 single cell; fix decoder if no learning.  
5. Sparse WP-U2 → lock default recipe → WP-U3 width atlas.  
6. Optionally refresh MEMORY_1B and root PLAN.md status once existence is shown.

---

## 10. Changelog

| Date | Change |
|------|--------|
| 2026-08-05 | Initial experiment plan on `plan/unary-swarm` |
| 2026-08-05 | WP-U0–U5 code harnesses landed (`v0_8` … `v0_12`) |
| 2026-08-06 | CPU pure-wall ladder finished; notes in `experiments/v0_*_NOTES.md` and `experiments/UNARY_LADDER_RESULTS.md` |
