# Binary Optimizers — Design, Reasons, and Trade-offs

Unified reference for optimizers in `binary_optimizers/optimizers/`.  
Historical notebook notes and later experimental variants are merged here; there is **no old/new split** — only families, mechanisms, and when to use each.

**Related:** fit-scale numbers in [`FIT_TRAINING_ANALYSIS.md`](FIT_TRAINING_ANALYSIS.md), roadmap in [`IMPROVEMENT_PROPOSALS.md`](IMPROVEMENT_PROPOSALS.md).

---

## Active vs paused experiments

Code for **all** optimizers remains importable. Default **fit** / **sweep** run the **active** set.

| Set | Count | Optimizers |
| :--- | :---: | :--- |
| **Active Bit-MLP optimizers** | **9** | `ema_flip`, `adam`, `signum`, `cosine_voting`, `ste`, `sparse_sign`, `voting`, `threshold_if`, `hybrid_accumulator` |
| **Active swarm combinations** | **4** | `swarm_mlp` × (`swarm`, `swarm_log`, `swarm_log_dynamic`, `adam`) |
| **Paused** | **1** | `hybrid_v2` |
| **Package only** | several | `integrate_fire*`, bitlogic/rank, `logic_optimizer`, … |

```bash
# Full comparison (Bit-MLP suite + swarm paths; cache hits when possible)
uv run python experiments/run_fit_training.py --include-swarm
# Swarm only
uv run python experiments/run_fit_training.py --swarm-only
```

Re-enable a paused name:

```bash
uv run python experiments/run_fit_training.py --optimizers hybrid_v2 --epochs 15
```

### Potential among re-enabled research opts

| Optimizer | Why still enabled | Potential |
| :--- | :--- | :--- |
| **`hybrid_accumulator`** | Adaptive fire-rate IF (unique control loop) | Event/energy metrics; not top MNIST FC accuracy yet |
| **`threshold_if`** | Clean IF baseline | Soft-fire / threshold tuning; hardware IF story |
| **`voting`** | Consensus + push (≠ Signum) | Slower, vote-buffered sign changes |

**Still paused:** `hybrid_v2` — stack with little unique upside vs Signum/Cosine unless used for ablations.

---

## Recommended default (Bit-MLP / MNIST-style STE nets)

| Setting | Recommendation |
| :--- | :--- |
| **Default optimizer** | **`ema_flip`** (`EMAFlipOptimizer`) |
| **Strong alternatives** | `signum`, `cosine_voting`, or `adam` (higher train memory) |
| **BN / bias** | Dual training: binary opt on 2D weights + Adam on 1D (fit pipeline) |
| **Deploy** | Extract ±1 / bitpacked weights; inference cost is independent of training optimizer |

### Why `ema_flip`

On fit-scale full MNIST (Bit-MLP h=128, 15 epochs, seed 42, dual BN-Adam):

| Optimizer | Best test | Final test | Packed (±1) | Train mem (approx.) |
| :--- | ---: | ---: | ---: | ---: |
| **ema_flip** | **96.26%** | **96.14%** | **96.14%** | ~795 KB |
| adam | 96.09% | 94.78% | 94.78% | ~1.19 MB |
| signum | 96.03% | 95.78% | 95.78% | ~795 KB |
| cosine_voting | 95.77% | 95.77% | 95.77% | ~795 KB |

- **Beats Adam on best and final** accuracy in this protocol while using **one** momentum-style buffer instead of Adam’s two moments.
- **Holds the plateau**: peak ≈ final (almost no late decay). Adam peaks mid-training then drops ~1.3 pp by the last epoch.
- **Deployment-friendly**: packed (±1) accuracy matches STE forward for this model.

### Trade-offs of the default

| Dimension | EMAFlip | vs Adam | vs pure Signum |
| :--- | :--- | :--- | :--- |
| Accuracy (fit MNIST) | Highest in suite | Slightly better final | Slightly better |
| Train optimizer memory | One EMA buffer | **Lower** | Similar |
| Hyperparameters | `lr`, `lr_min`, `momentum`, `threshold_scale`, `total_steps` | More knobs than Adam’s lr | Extra adaptive gate |
| Early epochs | Can lag (gate warm-up) | Adam often stronger ep 1–5 | Signum often stronger early |
| Evidence strength | Strong on this protocol | — | Multi-seed confirmation still recommended |

**When not to default to EMAFlip:** swarm population layers (use swarm / swarm-log opts); pure event-driven / neuromorphic studies (use integrate-and-fire family); ablations that require vanilla SGD-STE or Adam baselines.

**CLI / experiments:** fit training lists `ema_flip` first among binary specialists; full suite still includes baselines for comparison.

```bash
uv run python experiments/run_fit_training.py --optimizers ema_flip
# or full suite (cached after first run)
uv run python experiments/run_fit_training.py
```

---

## Shared design goals

Across the suite:

1. **Stability for binary / sign weights** — reduce flip-flapping from noisy per-batch gradients.  
2. **Less optimizer state than Adam** where possible — fewer auxiliary buffers.  
3. **Sign- and bit-centric updates** — prefer bounded direction signals over raw gradient magnitudes.  
4. **Compatible with STE forward** — latent floats (or swarm bits) with sign/binary forward and clamp.  
5. **Optional path toward discrete / logic-like rules** — fire, rank, or population flips.

**Training protocol note:** Fit experiments typically optimize **2D weights** with the binary optimizer and **BatchNorm/bias** with a small Adam (`DualOptimizer`). That is part of the accuracy story for binary methods vs pure Adam-on-all-params.

---

## Fit-scale comparison (single view)

Full MNIST, Bit-MLP, 15 epochs (see `FIT_TRAINING_ANALYSIS.md` for curves).

| Tier | Optimizers | Best test | Typical behavior |
| :--- | :--- | ---: | :--- |
| **S** | `ema_flip` | ~96.3% | Late surge, holds plateau |
| **A** | `adam`, `signum`, `cosine_voting` | ~95.8–96.1% | Strong fit |
| **B** | `ste` | ~94.5% | Stable, lower ceiling |
| **C** | `sparse_sign`, `threshold_if` | ~88–90% | Incomplete fit |
| **D** | `voting`, `hybrid_v2`, `hybrid_accumulator` | ~78–86% | Stuck / high variance |

**Takeaway:** Continuous **confidence-gated sign** updates (EMAFlip, Signum, CosineVoting) outperform **accumulate-then-fire** hybrids on this model class.

---

## Family overview

| Family | Keys / classes | Core idea |
| :--- | :--- | :--- |
| STE / float | `ste`, `adam` | Latent floats + clamp or Adam moments |
| Sign / voting | `voting`, `signum`, `cosine_voting`, `ema_flip`, `sparse_sign` | Directional votes, often with momentum |
| Integrate-and-fire | `threshold_if`, `integrate_fire*`, `hybrid_accumulator`, `hybrid_v2` | Accumulate pressure, fire discrete updates |
| Logic / rank | `bitlogic`, rank / adaptive / momentum-rank | Stochastic or top-k bit flips |
| Integer / log | `logic_optimizer`, `swarm_log_optimizer` | Integer counters / log-style flips |
| Swarm | `swarm`, swarm-log | Population of bits per weight |

---

## STE family

### `STEOptimizer` (`ste`)

- **Mechanism:** SGD step, then clamp 2D weights to `[-1, 1]`.  
- **Reason:** Classic straight-through baseline: standard backprop on latent floats, sign in forward (via layers).  
- **Pros:** Simple, well-understood, works with any STE layer.  
- **Cons:** No vote memory; can be noisy; lower ceiling than Signum/EMAFlip on fit MNIST (~94.5%).  
- **Use when:** Baseline comparisons, simplest binary training path.

### Adam (`adam`)

- **Mechanism:** Full Adam on all parameters.  
- **Reason:** Strong float optimizer upper bound for the same architecture.  
- **Pros:** Fast early progress, high peak accuracy.  
- **Cons:** ~2× optimizer-state memory; **late decay** on fit MNIST (best 96.1% → final 94.8%).  
- **Use when:** Accuracy ceiling reference, not memory-constrained training.

---

## Sign / voting family

### `VotingOptimizer` (`voting`)

- **Mechanism:** Accumulate signed batch consensus; push latent weights toward `sign(accumulator)` with `push_rate`.  
- **Reason:** Stabilize flips by requiring multi-batch consensus instead of reacting to one batch.  
- **Pros:** Interpretable consensus dynamics; bounded accumulator.  
- **Cons:** Sensitive to `push_rate`; fit MNIST shows **high test variance** and weaker final accuracy (~86% best / ~83% final with dual BN-Adam).  
- **Use when:** Studying consensus dynamics; not the accuracy default.

### `MomentumVotingOptimizer` / Signum (`signum`)

- **Mechanism:** Momentum buffer on gradients; update `p -= lr * sign(buf)`; clip weights. Optional confidence threshold.  
- **Reason:** Fix SignSGD drift (clip) and noise (momentum) while keeping **one** buffer.  
- **Pros:** Strong accuracy (~96% best); memory-efficient; simple.  
- **Cons:** Fixed lr (no built-in anneal); mild late noise.  
- **Use when:** Strong default alternative to EMAFlip; simpler hyperparameter surface.

### `CosineVotingOptimizer` (`cosine_voting`)

- **Mechanism:** Same as momentum sign updates, with **built-in cosine LR** from `lr_max` → `lr_min` over `total_steps` (optional restarts).  
- **Reason:** Exploration early, refinement late, without an external scheduler.  
- **Pros:** Stable late fit on MNIST (~95.8% final = best); good memory; no separate LR schedule code.  
- **Cons:** Must set `total_steps` ≈ real train length; aggressive restarts can destabilize (prefer `restart_period=0` for long fits).  
- **Use when:** You want Signum-like dynamics with automatic annealing.

### `EMAFlipOptimizer` (`ema_flip`) — **default for Bit-MLP**

- **Mechanism:** EMA of gradient; adaptive threshold = `threshold_scale * running_mean(|EMA|)`; update only confident weights with `-lr * sign(EMA)`; optional cosine `lr → lr_min`; optional true flip mode.  
- **Reason:** Replace expensive top-k ranking with an O(n) adaptive gate; reduce oscillation vs raw sign; keep Signum-level memory.  
- **Pros:** Best fit accuracy in suite; holds plateau; lower memory than Adam; packed = STE.  
- **Cons:** Early warm-up lag; more hyperparameters (`threshold_scale`, anneal horizon); multi-seed validation still wise.  
- **Use when:** Default training for STE Bit-MLP / similar fully connected binary nets.

### `SparseSignOptimizer` (`sparse_sign`)

- **Mechanism:** Momentum sign updates on a random (or magnitude-biased) subset; density holds at `density_start` for `density_hold_frac` of training, then anneals to `density_end`; optional cosine LR.  
- **Reason:** Weight-space dropout / cheaper steps; hardware-friendly sparse updates in principle.  
- **Pros:** Same memory as Signum; **fully fits** Bit-MLP when dense long enough (40-ep max-fit: **best 97.1%**, train 98% — see `SPARSE_SIGN_MAX_FIT.md`).  
- **Cons:** At short budgets (15 ep) with early sparsity, **under-fits** (~90% test); needs more epochs or high `density_hold_frac`; CPU masks add overhead without sparse kernels.  
- **Use when:** Longer train budgets; ablations on sparsity. For short runs prefer `ema_flip`. Max-fit recipe: `--optimizers sparse_sign --epochs 40`.

---

## Integrate-and-fire family

### `ThresholdedIntegrateFireOptimizer` (`threshold_if`)

- **Mechanism:** Accumulator decays and integrates anti-grad; when `|acc| > threshold`, set weight to `sign(acc)` and reset.  
- **Reason:** Event-driven flips only after sustained evidence.  
- **Pros:** Discrete, interpretable fire events; moderate memory (one acc buffer).  
- **Cons:** Fixed threshold hard to tune across layers; fit MNIST ceilings ~88% — often under-fires vs continuous sign methods.  
- **Use when:** Neuromorphic / event-driven experiments.

### `IntegrateFireOptimizer` / `AdaptiveIntegrateFireOptimizer`

- **Mechanism:** Integrate-and-fire on (often swarm) tensors; adaptive variant retunes threshold.  
- **Reason:** Same family with different parameterization and swarm focus.  
- **Pros / cons:** Same structural trade-offs as threshold IF; prefer for swarm-shaped params when studying fire rates.

### `HybridAccumulatorOptimizer` (`hybrid_accumulator`)

- **Mechanism:** EMA of grad + accumulator + **per-tensor** adaptive threshold; soft or hard fire toward `sign(acc)`.  
- **Reason:** Combine momentum smoothing with IF and self-tuning fire rate.  
- **Pros:** Rich state; theoretically noise-filtered.  
- **Cons:** Two buffers (~Adam-level memory); fit MNIST **stuck ~78%** — fire path under-updates vs continuous sign.  
- **Use when:** Research on adaptive fire rates; not accuracy-first training.

### `HybridV2Optimizer` (`hybrid_v2`)

- **Mechanism:** Stack: momentum EMA + cosine LR + optional sparse mask + optional soft IF fire.  
- **Reason:** Ablation-friendly combination of the strongest ideas.  
- **Pros:** Configurable (`use_fire`, `density`, …) to isolate components.  
- **Cons:** Default fire path plateaus ~82% on fit MNIST; continuous sign path (`use_fire=False`) is the promising ablation.  
- **Use when:** Structured ablations; not the production default until the sign path is validated.

---

## Logic, rank, and integer variants

### BitLogic / Rank / Adaptive / MomentumRank (`bitlogic.py`)

- **Mechanism:** Stochastic flip probability from grad pressure; or top-k “wrongness” flips; adaptive / momentum-rank refinements.  
- **Reason:** Move toward comparator / ranking rules with less continuous accumulation.  
- **Pros:** Explicit discrete flip control; rank-based determinism options.  
- **Cons:** Sensitivity / flip_rate tuning; top-k cost; mainly validated in notebook / swarm settings.  
- **Use when:** Logic-oriented or swarm-bit studies.

### `IntegerVotingOptimizer` (`logic_optimizer`)

- **Mechanism:** Integer accumulators, thresholded flips.  
- **Reason:** Avoid float application of gradients to binary weights.  
- **Pros:** Integer-friendly story.  
- **Cons:** Less exercised in the fit_v3 Bit-MLP suite.

### Swarm / SwarmLog (`swarm`, `swarm_log_optimizer`)

- **Mechanism:** Population of bits per weight; recruit/flip by pressure; log/integer variants.  
- **Reason:** Gradual discrete change via partial population flips; majority at inference.  
- **Pros:** Natural multi-bit representation; inference can collapse to majority (STE-sized).  
- **Cons:** Training memory × population size; needs swarm layers, not plain BitLinearSTE.

---

## Cross-cutting trade-offs

| Dimension | Prefer | Avoid / caution |
| :--- | :--- | :--- |
| **Accuracy (Bit-MLP MNIST fit)** | `ema_flip`, `signum`, `cosine_voting` | `hybrid_*`, sparse-early, unstable `voting` |
| **Train memory** | Sign/EMA family (~1 buffer) | Adam, hybrid (2 buffers) |
| **Inference memory** | Same after extract (bitpacked) | — (optimizer-independent) |
| **Simplicity** | `ste`, `signum` | Adaptive hybrids |
| **LR schedule free** | `cosine_voting`, `ema_flip` (built-in anneal) | Fixed-lr Signum/STE without external schedule |
| **Event-driven semantics** | IF family | Continuous sign defaults |
| **Swarm models** | `swarm` / swarm-log | Plain EMAFlip on 3D population without care |

**Inference:** After training, weights are extracted to ±1 / uint8 packing. Choice of training optimizer affects **accuracy of the final bits**, not the bitpacked size.

---

## Practical recipes

```text
Bit-MLP (MNIST / similar STE FC)
  → ema_flip  (default)
  → signum or cosine_voting as backups
  → adam as float upper bound

Need built-in LR anneal only
  → cosine_voting  (set total_steps = epochs * steps_per_epoch)

Event-driven / IF research
  → threshold_if or hybrid_accumulator (expect lower MNIST FC accuracy)

Sparse-update research
  → sparse_sign with density≈1 until fit, then anneal

Swarm layers
  → swarm / swarm_log optimizers + majority extract at inference
```

### Reproduce fit ranking

```bash
uv run python experiments/run_fit_training.py          # train missing fingerprints only
uv run python experiments/run_benchmark_checkpoints.py # eval saved nets, no train
```

Checkpoints live under local `checkpoints/` (gitignored).

---

## Module map

| Module | Main classes |
| :--- | :--- |
| `ste.py` | `STEOptimizer` |
| `voting.py` | `VotingOptimizer` |
| `signum.py` | `MomentumVotingOptimizer` |
| `cosine_voting.py` | `CosineVotingOptimizer` |
| `ema_flip.py` | `EMAFlipOptimizer` |
| `sparse_sign.py` | `SparseSignOptimizer` |
| `threshold_if.py` | `ThresholdedIntegrateFireOptimizer` |
| `integrate_fire.py` | `IntegrateFireOptimizer`, `AdaptiveIntegrateFireOptimizer` |
| `hybrid_accumulator.py` | `HybridAccumulatorOptimizer` |
| `hybrid_v2.py` | `HybridV2Optimizer` |
| `bitlogic.py` | BitLogic / Rank / Adaptive / MomentumRank |
| `logic_optimizer.py` | `IntegerVotingOptimizer` |
| `swarm.py` | `SwarmOptimizer` |
| `swarm_log_optimizer.py` | `SwarmLogOptimizer` |

Import surface: `binary_optimizers.optimizers`.
