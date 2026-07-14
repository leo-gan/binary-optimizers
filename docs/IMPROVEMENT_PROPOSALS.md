# Research: Fit-v3 Results → Improvement Proposals

**Source:** `results/fit_training.json` (tag `fit_v3`, schema 3), full MNIST, Bit-MLP (h=128), 15 epochs, seed 42, BN dual-Adam for non-Adam optimizers.  
**Protocol caveats:** single seed; one model size; STE train forward; packed eval = sign(weights).

---

## 1. Executive findings

| Tier | Optimizers | Best test | Role |
| :--- | :--- | ---: | :--- |
| **S** | `ema_flip` | **96.26%** | Best overall; beats Adam |
| **A** | `adam`, `signum`, `cosine_voting` | 95.8–96.1% | Strong fit; small gaps |
| **B** | `ste` | 94.5% | Solid, under-updates vs sign family |
| **C** | `sparse_sign`, `threshold_if` | 88–90% | Incomplete fit |
| **D** | `voting`, `hybrid_v2`, `hybrid_accumulator` | 78–86% | Stuck / thrash |

**Headline:** With BN-Adam dual training, **EMAFlip is the recommended default binary trainer**. It wins 5/5 baselines on best test, holds final ≈ best (almost no late decay), and matches packed accuracy to STE.

**Adam’s weakness:** peaks at epoch 11 (96.09%) then drifts to 94.78% final — classic late overshoot. EMAFlip and CosineVoting **hold** the plateau better.

**Memory:** top binary opts use ~795 KB train mem (one momentum buffer); Adam ~1.19 MB (two moments). Hybrid family also ~1.19 MB (EMA + accumulator).

---

## 2. Curve signatures (what the data says)

### 2.1 Winners: late surge + hold

```
ema_flip:   slow start (~87%) → steady climb → peak 96.3% ep13 → hold 96.1%
cosine:     noisy early → strong anneal finish 95.8% (peak = final)
signum:     strong early (~93%) → peak 96.0% ep10 → mild noise, ends 95.8%
```

**Implication:** Adaptive gating + LR decay (EMAFlip) or pure cosine (CosineVoting) both work; **continuous sign steps on confident weights** beat discrete fire for MNIST Bit-MLP.

### 2.2 Adam: early king, late drift

```
adam: 0.92 → 0.96 by ep11 → 0.948 final (−1.3 pp late decay)
```

**Implication:** Binary nets may need **weight decay / freeze BN / lower late lr** even for Adam; or early-stop on best checkpoint (not last epoch).

### 2.3 Mid tier: under-capacity of update rule

```
sparse_sign: oscillates 0.82–0.90, train_final 0.87 < test sometimes → not fully fitting
threshold_if: hard fire → sticky 0.85–0.88; never enters 0.94 regime
```

**Implication:** Sparse under-updates (even with density anneal); IF threshold too conservative relative to continuous sign SGD.

### 2.4 Failures: stuck plateaus

```
hybrid_v2:          monotone crawl 0.76→0.82 — IF soft fire under-powered vs signum-scale updates
hybrid_accumulator: chaotic mid-training (0.55 dips); never escapes ~0.78
voting:             high variance test (0.67–0.86); push_rate dynamics unstable with BN-Adam
```

**Implication:** Stacking IF on top of momentum **without** matching Signum’s per-step sign magnitude loses ~14 pp. Hybrid_v2’s `use_fire=True` path integrates small pressure then soft-blends — effective step size ≪ `lr * sign(ema)`.

---

## 3. Root-cause hypotheses (ranked by evidence)

| # | Hypothesis | Evidence |
|:-:| :--- | :--- |
| H1 | **Continuous confident sign updates > accumulate-then-fire** for this model | EMAFlip/Signum/Cosine ≫ Hybrid/IF |
| H2 | **BN dual-Adam is necessary but not sufficient** | All dual-trained; hybrids still fail → binary update rule is the bottleneck |
| H3 | **Late LR decay preserves peaks** | EMAFlip/Cosine hold; Adam decays without binary-specific schedule |
| H4 | **Sparsity hurts full fit** unless density stays near 1.0 until train acc > 0.95 | Sparse peaks 89.7% with train only 87% |
| H5 | **Hybrid_v2 fire threshold too high / soft_alpha too low** | Curve asymptotes ~81% (local regime) |
| H6 | **Single seed noise** | ±0.2–0.5 pp uncertainty; S-tier ranking of EMAFlip vs Adam may flip |

---

## 4. Proposed improvements

### P0 — Confirm and productize the winner (1–2 days)

| ID | Action | Why | Success metric |
| :--- | :--- | :--- | :--- |
| **P0.1** | Multi-seed (3×) EMAFlip vs Adam vs Signum vs Cosine | Guard against seed-42 luck | Mean±std best/final; EMAFlip mean ≥ Adam |
| **P0.2** | Save **best-epoch** weights in checkpoint (not only last) | Adam loses 1.3 pp last-vs-best; fair reporting | Report best & last; deploy best |
| **P0.3** | Default CLI / docs: recommend `ema_flip` for Bit-MLP | Analysis already shows S-tier | Docs + `FIT_OPTIMIZERS` order — **DONE** (`docs/optimizers.md`, `--default-only`) |

### P1 — Close residual gaps / fix broken families (high impact)

| ID | Target | Change | Rationale from curves |
| :--- | :--- | :--- | :--- |
| **P1.1** | `hybrid_v2` | Ablation matrix: `use_fire∈{F,T}`, `density∈{1.0,0.9}`, `soft_alpha∈{0.5,1.0}`, `threshold∈{0.02,0.04}` | Isolate whether fire is the failure; expect `use_fire=False` ≈ Signum/Cosine |
| **P1.2** | `hybrid_v2` | **HybridV2-fast path:** if `use_fire=False`, match Signum update `p += -lr * sign(ema)` with cosine lr (already partially there — tune lr_max=0.05–0.1) | Currently stuck at 82% |
| **P1.3** | `sparse_sign` | Keep density=1.0 until train_acc ≥ 0.95 or epoch ≥ 10, then anneal to 0.5 | Early sparsity blocks fit (train 87%) |
| **P1.4** | `threshold_if` | Lower threshold 0.02→0.01; or fire with soft blend like Hybrid soft_fire | Escape 88% ceiling |
| **P1.5** | `voting` | Replace fixed push_rate with Signum-style momentum sign; keep accumulator only as optional confidence | Voting alone is D-tier with BN-Adam |

### P2 — Push past 96.5% / beat Adam more clearly

| ID | Action | Rationale |
| :--- | :--- | :--- |
| **P2.1** | **EMAFlip+**: threshold_scale schedule (0.5→0.25) so early updates freer, late more selective | Early ema_flip dips ep1–4 (0.84–0.87); late is excellent |
| **P2.2** | **EMAFlip + SWA / polyak** average of last 3 epochs | Hold plateau; small free gain |
| **P2.3** | **Label smoothing 0.05** or mild mixup | Small gen gap already; may help Adam more than EMAFlip |
| **P2.4** | **bit_mlp_large (h=256)** fit_v3 suite | Check whether ranking is capacity-dependent |
| **P2.5** | **Longer train 30–40 ep** for Cosine/EMAFlip only | Cosine still climbing at ep15 (peak=final) |

### P3 — Evaluation & deployment honesty

| ID | Action | Rationale |
| :--- | :--- | :--- |
| **P3.1** | Benchmark **true XNOR+popcount PackedModel** path (not only sign-weights STE) | Packed≈STE now; validate engine parity under load |
| **P3.2** | Latency + energy proxy at batch 1/8/32 for top-3 opts | Training mem already favors binary; inference story incomplete for ranking |
| **P3.3** | CIFAR-10 SmallBitConvNet smoke (5–10 ep) for EMAFlip vs Adam vs Signum | MNIST may overstate binary ease |

### P4 — Engineering / experiment hygiene

| ID | Action | Rationale |
| :--- | :--- | :--- |
| **P4.1** | Checkpoint stores `best_state_dict` + `last_state_dict` | Enables P0.2 without retrain |
| **P4.2** | Auto-generate this analysis section from curves (decay, stall flags) | Current diagnostics under-flag hybrid plateaus |
| **P4.3** | Ablation runner: `experiments/run_ablation_hybrid_v2.py` with cached fingerprints per flag set | Make H1/H5 falsifiable cheaply |

---

## 5. Suggested implementation order

```text
Week 1
  P0.2 best-epoch checkpoint
  P1.1–P1.2 HybridV2 ablations (use_fire=False likely promotes it to A-tier)
  P1.3 SparseSign late-only sparsity
  P0.1 3-seed confirm on top-4

Week 2
  P2.1 EMAFlip+ threshold schedule
  P2.4 large MLP
  P3.1–P3.2 inference bench on saved top-3 checkpoints only (no retrain)
```

**Do not prioritize:** more HybridAccumulator tuning before HybridV2 `use_fire=False` ablation — same failure mode, worse numbers.

---

## 6. Concrete “next patch” (minimal code)

1. **`HybridV2` default `use_fire=False`** (or add `hybrid_v2_sign` config) with `lr_max=0.06` cosine — expect ~95%+ if H1 holds.  
2. **`SparseSign`**: `density_start=1.0`, `density_end=0.5`, but delay anneal: density = 1.0 for first `0.6 * total_steps`.  
3. **Checkpoints:** save best-by-test epoch weights; benchmark loads best by default.  
4. **3 seeds** for `ema_flip`, `adam`, `signum`, `cosine_voting` only.

Each change bumps fingerprint (kwargs/schema) → only affected opts retrain; others stay cache hits.

---

## 7. Risk register

| Risk | Mitigation |
| :--- | :--- |
| EMAFlip win is seed-specific | P0.1 multi-seed |
| MNIST not representative | P3.3 CIFAR smoke |
| Dual BN-Adam confounds “binary purity” | Report binary-only ablation (bn_adam_lr=0) for paper honesty |
| HybridV2 name oversells | Rename or demote until ≥90% |

---

## 8. Bottom line

| Keep | Fix | Drop / deprioritize |
| :--- | :--- | :--- |
| EMAFlip (+ BN-Adam) as default | HybridV2 fire path; SparseSign early density | HybridAccumulator until V2 sign-path works |
| CosineVoting as stable #2 binary | Voting instability | Aggressive cosine restarts (already disabled) |
| Checkpoint cache workflow | Best-epoch save | Full retrain of all opts each change |

**Primary research claim supported by fit_v3:**  
*Adaptive confidence-gated sign descent with cosine LR and BN-Adam matches or exceeds full Adam on MNIST Bit-MLP at lower optimizer-state memory, while integrate-and-fire hybrids under-update and plateau ~15 pp lower.*
