# Swarm optimizer — terminology, design, and trade-offs

This document explains **what “Swarm” means in this repository**, how it differs from
standard and STE-style training, what design choices we made, and why—plus the
trade-offs we accept.

It supports the project **grand goal**: **fully binary or fully ternary training** —
a discrete **network** and a discrete **optimizer**, without a floating-point master
weight \(W\). See the root [README](../README.md) and [PLAN.md](PLAN.md).

---

## 1. One-sentence idea

**Swarm** training stores each weight as a small **population of discrete pieces**
(agents or digits), and **updates those pieces with probabilistic discrete steps**
driven by gradient *pressure*, instead of running Adam/SGD on a full-precision
weight matrix.

---

## 2. Where Swarm sits relative to other optimizers

| Approach | What is stored as the weight | How it is updated |
|----------|------------------------------|-------------------|
| SGD / Adam | FP32 (or FP16) matrix \(W\) | Continuous steps on \(W\) |
| BitNet QAT / typical STE | FP **master** \(W\); forward uses discrete view \(\tilde W\) | Continuous steps on master \(W\); STE fakes grads through the discrete map |
| **Swarm (this project)** | **Only** discrete state (agents / bits / trits) | **Discrete** flips or integer Δ on that state |

**Latent-free** = there is no second “real” FP weight behind the discrete view. The
buffer you flip or re-encode **is** the parameter.

```
Standard / STE:     loss ← forward(discrete(W_fp)) ← update W_fp
Swarm (latent-free): loss ← forward(decode(population)) ← update population
```

Autograd may still compute floating-point gradients through a **soft STE path**
(e.g. a float view of bits/agents that requires grad). That pressure is used only to
decide discrete updates; it is **not** a master weight that lives across steps as
the true state.

---

## 3. Terminology (project dictionary)

Use these terms consistently in code, PROTOCOLs, and papers.

### 3.1 Network objects

| Term | Meaning |
|------|---------|
| **Weight** | One scalar entry of a weight matrix (one learnable link between two units). Prefer this over “connection.” |
| **Layer** | A linear (or similar) map; e.g. `Int8SwarmLinear`, `Int8CarrySafeLinear`. |
| **Population** | The discrete state tensor for a layer, shape roughly `[out, in, width]`. |
| **Effective weight** \(w\) | Scalar used in the matmul after decoding the population, usually scaled into \([-1,1]\), often times \(1/\sqrt{\mathrm{fan\_in}}\). |

### 3.2 Swarm / discrete state

| Term | Meaning |
|------|---------|
| **Agent** | One discrete item in the population for a single weight (project term; not standard BitNet jargon). In unary Swarm: one ±1 voter. In place-value coding: one bit or trit digit. |
| **Swarm** | The collection of agents (or the coding+update family) that represents and trains weights. Colloquially also the optimizer that steps that state. |
| **Width** | How much discrete state **per weight** (not “wider network”). Two different widths: |
| → **Population size** \(S\) | Unary family (v0.1): number of equal ±1 agents. |
| → **Register width** \(n\) | Place-value family (v0.2–v0.4): number of bits or trits. |
| **Coding** | *How* one weight is built from agents/digits (unary majority vs place-value vs exp/mant, …). |
| **Update** | *How* that discrete state is changed each step (independent flips vs carry-safe ±Δ on an integer, …). |

Coding and update are **orthogonal**. Confusing them leads to false claims like
“swarm size 16 is best” when you only tried one coding.

### 3.3 Optimizer dynamics

| Term | Meaning |
|------|---------|
| **Pressure** | Gradient-derived signal that says “this weight should move up or down.” Often mean/sum of STE grads over agents, then EMA-smoothed. |
| **Recruit rate** | Multiplier turning |pressure| into a flip/step probability: \(\mathrm{prob} \propto \mathrm{clamp}(|p|\cdot\mathrm{recruit\_rate})\). |
| **Flip** | Unary: an agent changes sign (+1 ↔ −1). Place-value: a bit/trit may change when the integer re-encodes. |
| **Carry-safe** | Update the **integer** \(v\) by ±Δ, then re-encode bits so carries are exact (no independent bit flips fighting place values). |
| **STE (manual)** | Straight-through estimator: forward uses hard discrete decode; backward flows through a soft float view of the population so each agent/digit gets a gradient. |

### 3.4 What “Swarm optimizer” is *not*

- Not a multi-agent RL system or particle-swarm global optimizer over network topology.  
- Not (by default) a subclass of `torch.optim.Optimizer` in the experiment ladder—the
  research implementations often step **buffers**, not `nn.Parameter` weights.  
  (There is also a historical `binary_optimizers.optimizers.swarm.SwarmOptimizer` that
  flips 3-D parameter tensors; experiment v0.1+ are the intentional latent-free line.)  
- Not the same as “any binary network.” Binary *inference* alone does not imply a Swarm
  *optimizer*.

---

## 4. Core loop (shared skeleton)

Regardless of coding, one training step looks like this:

1. **Forward**  
   - Materialize a float view of the discrete population (for STE).  
   - **Decode** effective \(w\) (majority / place-value / …).  
   - Matmul with optional fixed gain \(1/\sqrt{\mathrm{fan\_in}}\).  

2. **Loss + backward**  
   - Standard loss (e.g. cross-entropy).  
   - Grads attach to the soft population view → **per-agent (or per-digit) pressure**.  

3. **Swarm step** (no FP update of a master \(W\))  
   - Aggregate pressure over the swarm dimension (mean or sum, depending on coding).  
   - Optional **EMA** of pressure (inertia / noise smoothing).  
   - Map pressure → **probability** of a discrete change (and sometimes step magnitude).  
   - Sample and apply flips or integer Δ; write back to the **int8 population**.  

4. **Optional FP SGD**  
   - Only for **LayerNorm affine** parameters when `ln_mode=affine`—those are not
     claimed as binary state. Prefer `ln_mode=none` when studying pure discrete weights.

### Why probability (not hard threshold)?

Hard “always flip if |g| > τ” is brittle: scale of gradients depends on batch, depth,
and coding. A soft recruit probability with a **cap** (`max_flip_prob` / `max_step_prob`)
limits thrashing while still allowing rare large updates. EMA pressure acts like a
low-pass filter so single-batch noise does not flip half the net.

### Why STE at all?

Without some path for gradients into discrete variables, you get no learning signal.
STE is a **compromise**: we keep discrete **state**, but allow an FP **signal** for
direction. That is intermediate between pure combinatorial search and full FP training.
Fully integer backward remains a longer-term ambition, not the current requirement.

---

## 5. Coding families (how one weight is represented)

### 5.1 Unary majority (v0.1) — “committee of equal voters”

- **Alphabet:** each of \(S\) agents is \(+1\) or \(-1\) (int8).  
- **Decode:** effective weight from majority / STE through normalized sum  
  \(\mathrm{sum}/S \in [-1,1]\) (forward often uses sign of the sum).  
- **Update:** flip agents that disagree with \(-\mathrm{sign}(\mathrm{pressure})\) with
  probability from |pressure|.  

**Intuition:** one weight is a ballot. Capacity is “how many voters,” not “how many
binary digits of an integer.”

**Trade-offs**

| Pros | Cons |
|------|------|
| Simple mental model; robust majority smoothing | Memory \(\propto S\) per weight; large \(S\) is slow |
| STE assigns equal credit to agents | Does not natively build multi-scale / place-value structure |
| Empirically improves with \(S\) until a plateau | Plateau needs large \(S\) (e.g. hundreds) for best MNIST sketch accuracy |

**WP1 sketch:** accuracy rises strongly \(S=8\to128\), soft plateau \(\approx 256\)–\(1024\).

### 5.2 Place-value binary register (v0.2 → v0.3)

- **Alphabet:** bits \(b_i \in \{0,1\}\), length \(n\).  
- **Decode (fixed-point):**  
  \(v = \sum_i b_i 2^i\),  
  \(w = 2v/(2^n-1) - 1 \in [-1,1]\).  
- **Update (v0.3 carry-safe):** change integer \(v\) by ±Δ (probabilistic, size from
  |pressure|), clamp to \([0, 2^n-1]\), **re-encode** bits.  

**Intuition:** one weight is a small unsigned integer, then affine-mapped to \([-1,1]\).

**Why carry-safe beats independent bit flips (reasoning)**

Independent flips on place-value bits are **misaligned** with the decode: flipping a
high bit is a huge jump in \(w\); flipping a low bit is tiny. Random independent flips
do not respect “move \(v\) by one notch.” Carry-safe updates change **\(v\)** first, so
the discrete geometry matches the continuous \(w(v)\). Empirically on MNIST, v0.3
clearly beat v0.2-style non-carry-safe place-value under comparable settings.

**Trade-offs**

| Pros | Cons |
|------|------|
| Compact multi-level weights with \(n\) bits | Must scale Δv with \(v_{\max}\) when sweeping \(n\) or wide registers barely move |
| Carry-safe steps match fixed-point geometry | int64-safe \(n \le 62\) for naive integer \(v\) |
| Strong MNIST results at small \(n\) (peak \(\approx 8\)) | Very large \(n\) under fixed train defaults **hurt** (cliff ≥32 on our sketch) |

**WP1 sketch:** unimodal peak at **\(n \approx 8\)**; 16 close; 32–62 plateau ~0.89.

### 5.3 Balanced ternary place-value (v0.4)

- Digits in \(\{-1,0,+1\}\) with base-3 place values; natural **zeros**.  
- Carry-safe steps on the ternary integer.  

**Trade-offs:** richer alphabet and sparsity of zeros vs more complex encode/decode;
on the MNIST toy, accuracy was in the same ballpark as v0.3 (not a free win).

### 5.4 Encoding variants on a fixed budget (v0.6)

Same total bit count \(n\), different **structure**:

| Encoding | Idea | Sketch outcome (MNIST) |
|----------|------|-------------------------|
| **Fixed-point** | All bits are mantissa-like place values | **Best** |
| **Exp + mantissa** | Some bits scale, some fill significand (\|w\| still normalized) | Clearly worse; more exp bits worse still |
| **Block scale** | Shared scale per row + per-weight mantissa | Between fixed and exp/mant |

**Reasoning for fixed winning on our sketches:** MNIST-scale MLP + fan-in gain may
not need multi-scale weights; our exp path forces \(|w|\le 1\) and uses approximate
exp updates—so “float-like” may be under-specified or mismatched to the optimizer.
CIFAR flat-MLP probe (v0.7) **kept the same ranking** (fixed > exp/mant) at lower
absolute accuracy.

---

## 6. Update rules in more detail

### 6.1 Unary flip (v0.1)

\[
\begin{aligned}
p &\leftarrow \mathrm{EMA}(\nabla\text{-pressure}) \\
\pi &\leftarrow \mathrm{clamp}(|p|\cdot r_{\mathrm{recruit}},\, 0,\, \pi_{\max}) \\
\text{target} &\leftarrow -\mathrm{sign}(p) \\
\text{flip agent }a &\text{ if } a \ne \text{target and } U < \pi
\end{aligned}
\]

Only agents that **disagree** with the desired sign are candidates—avoids flipping
aligned agents by noise.

### 6.2 Carry-safe integer step (v0.3)

\[
\begin{aligned}
p &\leftarrow \mathrm{EMA}(\partial L/\partial w) \\
\Delta &\leftarrow \mathrm{round}(\mathrm{clamp}(|p|\cdot s_{\mathrm{step}},\, 1,\, \Delta_{\max})) \\
v &\leftarrow \mathrm{clamp}\bigl(v - \mathrm{sign}(p)\cdot\Delta,\, 0,\, v_{\max}\bigr) \quad \text{(with prob \(\pi\))} \\
\text{bits} &\leftarrow \mathrm{encode}(v)
\end{aligned}
\]

**Width fairness:** when sweeping \(n\), scale \(\Delta_{\max}\) (and step scale) so
\(\Delta v / v_{\max}\) stays comparable to a reference \(n\) (e.g. 16). Without that,
wide registers look “broken” only because they barely move.

### 6.3 Hyperparameters (conceptual roles)

| Knob | Role | Trade-off |
|------|------|-----------|
| `recruit_rate` | Sensitivity of flip/step probability to pressure | Too low → stuck; too high → thrash / noise |
| `max_flip_prob` / `max_step_prob` | Cap on how aggressive one step is | Stability vs speed of adaptation |
| `grad_momentum` (EMA) | Smooth pressure over steps | Lag vs noise rejection |
| `max_step` / `step_scale` | Size of integer Δ (register) | Must track \(v_{\max}\) when \(n\) changes |
| `ln_lr` | FP SGD for LN affine only | Orthogonal to discrete weight story |

Research stance: **do not** grid these per representation unless a coding is *broken*
without a minimal fix. Representation science varies coding/width, not lr tables.

---

## 7. Design principles (reasonings)

### 7.1 Discrete state is the product

If the goal is a binary/ternary **optimizer + NN**, the system of record must be
discrete. STE-with-master-\(W\) is a different product (good engineering, different claim).

### 7.2 Match geometry of coding and update

Place-value decode + independent bit flips fight each other. Carry-safe integer
steps restore consistency. **Unary** equal votes pair naturally with independent flips.

### 7.3 Width is not one number

\(S\) (voters) and \(n\) (digits) answer different questions. Empirical curves differ:
unary likes large \(S\); fixed-point register peaks at small \(n\). Never report “swarm
size” without naming the family.

### 7.4 Fair comparison needs fair train budget

Epoch length varies with \(S\) and \(n\). Epoch-only early stop **favors slow cells**
(more wall time per patience count). Prefer **pure wall-clock** budgets
([TRAIN_BUDGET.md](TRAIN_BUDGET.md)): same max wall and wall stall across cells.

### 7.5 Representation laws before scale

Discover width/encoding laws on small nets first, then ask whether optima **move**
(CIFAR / deeper nets). That path serves the grand goal better than multi-seed
STE vs Swarm scoreboards.

---

## 8. Trade-offs summary

### 8.1 Swarm vs FP optimizers

| | Swarm (latent-free) | Adam / SGD on FP \(W\) |
|--|---------------------|------------------------|
| Memory of weights | Can be int8 population | FP32/FP16 master |
| Update | Discrete, stochastic | Continuous, deterministic (usually) |
| Theory / tooling | Immature; custom step | Mature ecosystem |
| Dynamic range | Limited by coding (\(n\), \(S\)) | Full float range |
| Hardware story | Aligns with bit-serial / low-bit inference *if* train stays discrete | Train/infer mismatch common |

**1B-parameter static budgets** (weights + grads + optimizer; no activations): see
**[MEMORY_1B.md](MEMORY_1B.md)** — e.g. FP32+Adam ~16 GB, binary STE+SGD ~8 GB,
packed binary Swarm \(n{=}8\) ~5 GB, packed ternary Swarm \(n{=}8\) ~6 GB.

### 8.2 Swarm vs STE (with master \(W\))

| | Swarm | STE + master \(W\) |
|--|-------|---------------------|
| State after training | Discrete by construction | Discrete only if you quantize away \(W\) |
| Claim “binary trained” | Stronger | Weaker / ambiguous |
| Stability | Harder; depends on recruit + coding | Easier; FP master absorbs fine steps |
| Research focus | Coding, width, discrete dynamics | Often LR and quant schedule |

### 8.3 Unary vs register (within Swarm)

| | Unary (\(S\)) | Register (\(n\)) |
|--|---------------|------------------|
| Semantics | Equal voters | Place-value integer |
| Typical good width (MNIST sketch) | Large \(S\) (hundreds) | Small \(n\) (~8) |
| Cost | Linear in \(S\) | Linear in \(n\) (usually smaller) |
| Natural multi-level \(w\) | Via majority fraction | Via many \(v\) levels |
| Failure mode | Too small \(S\) → noisy / weak | Too large \(n\) → hard to train under fixed defaults |

### 8.4 Fixed-point vs exp/mant (within register budget)

| | Fixed-point | Exp + mantissa (our designs) |
|--|-------------|------------------------------|
| MNIST / CIFAR flat MLP sketch | Wins | Loses; worse with more exp bits |
| Complexity | Low | Higher (split, bias, scale norm) |
| When exp might still win | Harder scales / free scale / better exp updates / rescue of large-\(n\) cliff | — |

---

## 9. Implementation map in this repo

| Artifact | Role |
|----------|------|
| `experiments/v0_1/` | Unary Swarm layers + `SwarmOptimizerV01` |
| `experiments/v0_2/` | Place-value coding, non-carry-safe lineage |
| `experiments/v0_3/` | **Reference** carry-safe register + `SwarmOptimizerV03` |
| `experiments/v0_4/` | Ternary place-value + carry-safe |
| `experiments/v0_5_width_*` | Width atlas (Question A) |
| `experiments/v0_6_encoding/` | Encoding atlas (Question B) |
| `experiments/v0_7_cifar_encoding/` | Sparse CIFAR fixed vs exp/mant |
| `binary_optimizers/optimizers/swarm.py` | Older Parameter-based flip Swarm |
| `binary_optimizers/training/budget.py` | Pure wall train protocol |
| `binary_optimizers/store/` | DuckDB runs + experiment version ids |

---

## 10. Open questions (honest)

1. **How much FP signal can we remove** while still training (integer backward, discrete
   norms/acts)?  
2. **Do width/encoding optima move** on real convnets / larger models (WP3 beyond
   flat CIFAR MLP)?  
3. **Can exp/mant be redesigned** so multi-scale coding helps when fixed-point cliffs?  
4. **Hardware cost model:** is unary large-\(S\) ever preferable to small-\(n\) registers
   under real memory/bandwidth constraints?  
5. **Theory:** convergence for stochastic discrete steps with STE pressure is thin;
   empirics lead for now.

---

## 11. Practical guidance

| If you want… | Prefer |
|--------------|--------|
| Default latent-free baseline | **v0.3** carry-safe binary, \(n\approx 8\), pure wall train |
| Study population size | **v0.1** unary; sweep \(S\), not \(n\) |
| Study encoding structure | Fixed total \(n\); compare fixed vs exp/mant (v0.6/v0.7) |
| Ternary zeros / trit structure | **v0.4** |
| Compare to STE with master \(W\) | `experiments/ste_vs_swarm` (secondary to the grand goal) |

**Do not** mix unary \(S\) and register \(n\) in one “swarm size” plot without labeling
coding and update.

---

## 12. Related docs

| Doc | Content |
|-----|---------|
| [../README.md](../README.md) | Grand goal and project map |
| [PLAN.md](PLAN.md) | Research roadmap (WP1–WP3) |
| [OPTIMA_STATUS.md](OPTIMA_STATUS.md) | Empirical width/encoding/CIFAR rankings |
| [TRAIN_BUDGET.md](TRAIN_BUDGET.md) | Why pure wall budgets |
| [EXPERIMENT_VERSIONS.md](EXPERIMENT_VERSIONS.md) | Protocol-bump experiment IDs |
| [optimizers.md](optimizers.md) | Historical notebook optimizers (STE, voting, …) |

---

## 13. Glossary (quick reference)

- **Agent** — discrete unit of state for one weight.  
- **Coding** — decode map population → \(w\).  
- **Update** — discrete rule that changes population.  
- **Latent-free** — no FP master \(W\).  
- **Pressure** — grad-derived direction/magnitude signal.  
- **Recruit** — probability of taking a discrete step.  
- **Carry-safe** — step integer, re-encode digits.  
- **Width** — \(S\) or \(n\) per weight, not network width.  
- **Swarm optimizer** — discrete stepper for population-backed weights under STE pressure.
