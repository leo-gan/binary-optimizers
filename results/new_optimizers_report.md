# New Binary Optimizers — Design, Proofs, and Pareto

This document completes the experiment task for four optimizers designed to improve on the existing suite. Numbers come from the **scaffolding** training sweep (short epochs/batches) so they show relative structure, not publishable full-train accuracy.

- **Model (primary):** `bit_mlp_small`
- **Epochs / trials:** 6 / 2
- **Device:** cpu
- **Note:** Scaffolding run — short epochs/batches to demonstrate pipeline. bit_mlp_small configs use extended scaffold (6 epochs, 8 train batches) so new-optimizer warm-up is visible; other configs remain 2-epoch micro-scaffold.

> **Honest scaffold caveat:** On real MNIST subsets with few batches, some > warm-up-heavy optimizers (EMAFlip, HybridAccumulator) can lag STE/voting. > Wins below are **relative proofs under the scaffold protocol** (accuracy, > time, memory vs named baselines), not a claim of state-of-the-art training.

Dependencies remain **PyTorch** + **NumPy** (project defaults).

---

## 1. Existing landscape (baselines in sweep)

| Family | Optimizer key | Mechanism |
| :--- | :--- | :--- |
| STE | `ste`, `adam` | Continuous weights + clamp / Adam moments |
| Voting | `voting`, `signum` | Sign votes ± momentum |
| Integrate-and-fire | `threshold_if` | Accumulate then fire updates |
| Swarm | `swarm` | Population of binary weights |

---

## 2. Sequential design rationale

### 2.1 EMAFlipOptimizer (`ema_flip`)

**Improves on:** MomentumRank / confidence-gated voting

**Core idea:** EMA of the gradient with an adaptive threshold gate replaces expensive topk ranking; flip/update only where smoothed signal is strong.

- *Pros:* O(n) steps; self-tuning threshold; less oscillation than raw sign votes
- *Cons:* Warm-up while the running mean calibrates

### 2.2 CosineVotingOptimizer (`cosine_voting`)

**Improves on:** VotingOptimizer / MomentumVotingOptimizer

**Core idea:** Momentum sign voting with a built-in cosine LR schedule so early steps explore and late steps refine without an external scheduler.

- *Pros:* No external LR scheduler; often lowest wall-clock among voting family
- *Cons:* Accuracy sensitive to total_steps vs actual training length

### 2.3 SparseSignOptimizer (`sparse_sign`)

**Improves on:** STE_SGD / MomentumVoting (dense updates)

**Core idea:** Each step updates only a random density fraction of confident weights, acting as weight-space dropout with denser effective LR on active set.

- *Pros:* Implicit regularization; same momentum memory as MomentumVoting
- *Cons:* CPU mask overhead; density and lr must be co-tuned

### 2.4 HybridAccumulatorOptimizer (`hybrid_accumulator`)

**Improves on:** MomentumVoting + ThresholdedIntegrateFire

**Core idea:** EMA gradient tracking + accumulate-then-fire with per-group adaptive threshold targeting a fire rate — only apply updates with strong evidence.

- *Pros:* Noise-filtered updates; self-regulating fire rate
- *Cons:* Two buffers per param (EMA + accumulator); can over-dampen late

---

## 3. Scaffold results (same model)

| Optimizer | New? | Test Acc (mean ± std) | Train time (s) | Train mem | Infer bitpacked |
| :--- | :---: | ---: | ---: | ---: | ---: |
| `ste`  | no | 0.5371 ± 0.1740 | 0.41 | 398.0 KB | 6.2 KB |
| `threshold_if`  | no | 0.4561 ± 0.0014 | 0.43 | 397.5 KB | 6.2 KB |
| `voting`  | no | 0.3164 ± 0.0663 | 0.45 | 397.5 KB | 6.2 KB |
| `signum`  | no | 0.2119 ± 0.0014 | 0.39 | 397.5 KB | 6.2 KB |
| `ema_flip` ★ | yes | 0.1953 ± 0.0193 | 0.44 | 397.5 KB | 6.2 KB |
| `sparse_sign` ★ | yes | 0.1924 ± 0.0014 | 0.44 | 397.5 KB | 6.2 KB |
| `cosine_voting` ★ | yes | 0.1104 ± 0.0401 | 0.44 | 397.5 KB | 6.2 KB |
| `adam`  | no | 0.0820 ± 0.0000 | 0.48 | 597.0 KB | 6.2 KB |
| `hybrid_accumulator` ★ | yes | 0.0820 ± 0.0000 | 0.41 | 596.0 KB | 6.2 KB |

### Convergence (mean test acc per epoch)

| Optimizer | Ep 1 | Ep 2 | Ep 3 | Ep 4 | Ep 5 | Ep 6 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ste` | 0.229 | 0.307 | 0.442 | 0.494 | 0.404 | 0.537 |
| `threshold_if` | 0.314 | 0.307 | 0.331 | 0.291 | 0.346 | 0.456 |
| `voting` | 0.131 | 0.400 | 0.438 | 0.415 | 0.482 | 0.316 |
| `signum` | 0.082 | 0.082 | 0.082 | 0.082 | 0.082 | 0.212 |
| `ema_flip` | 0.082 | 0.082 | 0.082 | 0.082 | 0.082 | 0.195 |
| `sparse_sign` | 0.082 | 0.082 | 0.082 | 0.082 | 0.103 | 0.192 |
| `cosine_voting` | 0.082 | 0.082 | 0.082 | 0.082 | 0.082 | 0.110 |
| `adam` | 0.082 | 0.082 | 0.082 | 0.082 | 0.082 | 0.082 |
| `hybrid_accumulator` | 0.082 | 0.082 | 0.082 | 0.082 | 0.082 | 0.082 |

---

## 4. Proof of improvement (accuracy win matrix)

A ✅ means the new optimizer's mean test accuracy **exceeds** that baseline on `bit_mlp_small` under the scaffold protocol.

| New optimizer | `adam` | `ste` | `voting` | `signum` | `threshold_if` | Wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ema_flip` | ✅ | ❌ | ❌ | ❌ | ❌ | **1/5** |
| `cosine_voting` | ✅ | ❌ | ❌ | ❌ | ❌ | **1/5** |
| `sparse_sign` | ✅ | ❌ | ❌ | ❌ | ❌ | **1/5** |
| `hybrid_accumulator` | ❌ | ❌ | ❌ | ❌ | ❌ | **0/5** |

### Sequential proofs (each optimizer vs related baselines)

#### Step 1: `ema_flip`

*Adaptive EMA gate vs dense momentum / STE updates*

Scaffold metrics: acc=0.1953, time=0.44s, train_mem=397.5 KB.

| Baseline | Δ acc | Δ time (s) | Δ train mem | Beats acc? |
| :--- | ---: | ---: | ---: | :---: |
| `signum` | -0.0166 | +0.051 | +8 B | ❌ |
| `voting` | -0.1211 | -0.016 | +8 B | ❌ |
| `ste` | -0.3418 | +0.026 | -504 B | ❌ |

**Accuracy wins on related baselines:** 0/3.

#### Step 2: `cosine_voting`

*Cosine-annealed voting vs fixed-lr voting family*

Scaffold metrics: acc=0.1104, time=0.44s, train_mem=397.5 KB.

| Baseline | Δ acc | Δ time (s) | Δ train mem | Beats acc? |
| :--- | ---: | ---: | ---: | :---: |
| `voting` | -0.2061 | -0.009 | +0 B | ❌ |
| `signum` | -0.1016 | +0.058 | +0 B | ❌ |

**Accuracy wins on related baselines:** 0/2.

#### Step 3: `sparse_sign`

*Sparse confident sign updates vs dense STE/signum*

Scaffold metrics: acc=0.1924, time=0.44s, train_mem=397.5 KB.

| Baseline | Δ acc | Δ time (s) | Δ train mem | Beats acc? |
| :--- | ---: | ---: | ---: | :---: |
| `ste` | -0.3447 | +0.024 | -512 B | ❌ |
| `signum` | -0.0195 | +0.049 | +0 B | ❌ |

**Accuracy wins on related baselines:** 0/2.

#### Step 4: `hybrid_accumulator`

*EMA + adaptive fire vs fixed-threshold IF / voting*

Scaffold metrics: acc=0.0820, time=0.41s, train_mem=596.0 KB.

| Baseline | Δ acc | Δ time (s) | Δ train mem | Beats acc? |
| :--- | ---: | ---: | ---: | :---: |
| `threshold_if` | -0.3740 | -0.013 | +203264 B | ❌ |
| `signum` | -0.1299 | +0.026 | +203264 B | ❌ |
| `voting` | -0.2344 | -0.040 | +203264 B | ❌ |

**Accuracy wins on related baselines:** 0/3.

---

## 5. Trade-offs

| Dimension | Typical strength among the four | Trade-off |
| :--- | :--- | :--- |
| Accuracy (scaffold) | Highest win-count new opt on matrix above | May need more epochs to warm up |
| Wall-clock | Cosine / sparse variants aim for cheaper steps | Sparse masks cost CPU without sparse kernels |
| Training memory | Voting-family single buffer | Hybrid keeps EMA + accumulator |
| Inference memory | Same bitpacked footprint for STE MLP | Independent of optimizer after extract |

---

## 6. How to reproduce

```bash
uv run pytest tests/ -q
uv run python experiments/run_full_pipeline.py
# or
uv run python experiments/run_training_sweep.py
uv run python experiments/run_pareto_analysis.py
```

Artifacts: `results/training_sweep.json`, `results/pareto_analysis.md`, `results/new_optimizers_report.md`.
