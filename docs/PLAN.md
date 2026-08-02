# Swarm / latent-free training — roadmap

**Branch:** `plan/swarm-roadmap`  
**Last updated:** 2026-07-31 (Coding column explained for v0.1–v0.4)  
**Status:** v0.1–v0.4 complete as *existence proofs*; next work is representation science, not tuning

This document is the living plan for latent-free binary/ternary Swarm optimizers
(BitNet-inspired layers, no FP master weights). Experiment folders use semantic
minor versioning: `experiments/v0_N/`.

**Terminology (preferred):** use **weight** for one entry of a weight matrix (one
learnable link between two units). Do **not** use “connection” for that idea.
An **agent** is one discrete item in the per-weight swarm (project term; not
standard BitNet jargon).

---

## 1. Goal (research claim)

Train networks whose **weight state is discrete** (binary agents / bits / trits),
updated by **Swarm rules** (votes, place-value registers, carry-safe ±Δ), so
training does **not** keep a full-precision latent \(W\) (unlike BitNet QAT / STE).

### 1.1 Research stance (current)

We are exploring the **whole idea** of Swarm representations—not polishing a recipe.

| Do now | Do **not** prioritize now |
|--------|---------------------------|
| Change **how weights are represented** (size of swarm / register, exponential coding) | Multi-seed error bars for ±0.5% claims |
| Look for **generic rules** (like scaling laws: how optimum depends on model/data size) | Hyperparameter fine-tuning for small leaderboard gains |
| Sweep wide ranges (8 … 1024+ agents/bits) to see **shape of the curve** | Head-to-head STE vs Swarm for small improvements |
| Ask *what encoding structure works* (exponent vs mantissa) | Exhaustive LayerNorm mode tables unless they change the representation story |

Baselines from v0.1–v0.4 are **reference points** (“this family can learn”), not a competition to shave points.

| Claim | In scope today |
|-------|----------------|
| No FP master weight for linear layers | Yes (v0.1–v0.4) |
| Discrete optimizer update (flips / integer Δ) | Yes |
| Representation laws for swarm width & encoding | **Next (main focus)** |
| Bit-only backward / integer LayerNorm | **No** |
| Trillion-param pretrain | **No** |

Background notes: `docs/temp/research.md`, `docs/temp/deep-research-report.md`.

---

## 2. What is done

### 2.1 Experiment ladder (MNIST, seed=42, best test acc)

| Ver | Path | Coding | Update | none | no_affine | affine |
|-----|------|--------|--------|------|-----------|--------|
| **v0.1** | `experiments/v0_1/` | Unary majority, int8 ±1, S=32 | Stochastic flips + grad EMA | 0.9239 | 0.9166 | 0.9188 |
| **v0.2** | `experiments/v0_2/` | Place \(2^i\), multi-level \(s_{\mathrm{norm}}\) | Indep. bit flips, soft LSB bias | 0.9289 | 0.9357 | 0.9365 |
| **v0.3** | `experiments/v0_3/` | Binary register \(v\in[0,2^n-1]\) | **Carry-safe** adaptive ±Δv | **0.9589** | **0.9723** | **0.9745** |
| **v0.4** | `experiments/v0_4/` | Balanced ternary, places \(3^i\) | Carry-safe adaptive ±Δs | **0.9551** | **0.9717** | **0.9737** |

**Takeaways (representation, not leaderboard)**

1. **How you encode and update** matters more than small protocol tweaks: carry-safe place-value (v0.3) beat independent flips (v0.2) by a clear margin.  
2. Unary majority (v0.1) and fixed-width binary registers (v0.3) are **different representation families**—both can learn; they are not the same object as “swarm size.”  
3. Ternary place-value (v0.4) is a third family; accuracy was similar to v0.3 on this toy, with natural zeros.  
4. We only ever tried a **few widths** (e.g. S=32 unary, n=16 bits, 10 trits). We do **not** know the role of width at 8, 64, 128, 1024, …

Numbers also in DuckDB `results/experiments.duckdb` when ingested.

#### What “Coding” means in the table above

**Coding** is *how one **weight** is represented as discrete pieces of state*, and *how
those pieces are turned into a number used in the matrix multiply*.

It is **not** the same as the **Update** column (how training changes that state).
The same coding family can pair with different updates (v0.2 and v0.3 both use
binary place-value-like storage; only v0.3 is carry-safe on the integer).

Think of coding as: **storage alphabet + how each piece is valued + decode formula**.

It is also **not** LayerNorm mode, not optimizer hyperparameters, and not layer width
(784→128→10). “Swarm size” alone is not the whole story: in v0.1 the budget is
population size \(S\); in v0.2–v0.4 it is how many **digits** \(n\) under a different coding.

##### v0.1 — Unary majority, int8 ±1, S=32

**Alphabet:** each of \(S\) agents (default 32) is only **+1 or −1** (stored as int8).

**Equal place values:** every agent counts the **same**. There is no “this agent is
worth 8.” That is what **unary** means here: like tally marks, not like binary digits.

**Decode (how the weight is used in the forward pass):**

1. Sum (or average) the agents.  
2. Take the **sign** of that sum (majority vote). Ties forced to +1.  
3. The effective weight is essentially **binary**: mostly **−1 or +1**, not a fine
   multi-level number.

So one weight is a **committee of equal voters**, not a binary integer. With 32 agents
the *effective* forward weight is still coarse (majority ±1), while the sum can take
many intermediate values used as a soft path for gradients (manual STE through the
sum / normalized sum).

**Intuition:** “32 equal ±1 chips; the weight’s direction is whatever the majority says.”

##### v0.2 — Place \(2^i\), multi-level \(s_{\mathrm{norm}}\)

**Alphabet:** still discrete ±1 (bit-like) units, length \(n\) (default **16**).

**Unequal place values:** unit index \(i\) is worth **\(2^i\)** (1, 2, 4, 8, …).
Low indices are fine steps; high indices are large steps.

**Decode:**

\[
s = \sum_i a_i \cdot 2^i,\qquad
s_{\mathrm{norm}} = s \Big/ \sum_j 2^j
\]

so \(s_{\mathrm{norm}}\) lies in a range like \([-1, 1]\). The forward pass uses this
**multi-level** value (not only majority ±1). Flipping a low bit changes \(w\) a little;
flipping a high bit changes it a lot.

**Intuition:** “The discrete state is a **binary number-like** object (signed place
values), not a ballot of equal votes.” First **exponential / place-value** coding in
the ladder.

**Note:** In v0.2, **updates** still flipped bits somewhat independently (with LSB bias).
The *coding* is place-value; the *update* was not yet fully carry-safe.

##### v0.3 — Binary register \(v \in [0, 2^n-1]\)

**Alphabet:** bits \(b_i \in \{0, 1\}\) (not ±1 agents). Length \(n\) (default **16**).

**Coding as a normal unsigned binary integer:**

\[
v = \sum_i b_i \cdot 2^i \in \{0, 1, \ldots, 2^n - 1\}
\]

**Decode to a weight in about \([-1, 1]\):**

\[
w = \frac{2v}{2^n - 1} - 1
\]

So \(v = 0\) → \(w = -1\), \(v = 2^n-1\) → \(w = +1\), middle → near 0. This is
**fixed-point-style** coding: all bits are place-value digits of one integer, then
scaled to a weight.

Compared to v0.2: same idea of **powers of two**, but storage is clean **0/1 bits**
and decode is “integer → scaled weight.” The **Update** column (carry-safe ±Δ on \(v\),
then re-encode bits) is separate from this coding definition.

**Intuition:** “Each weight is a small **binary counter** from 0 to \(2^n-1\), mapped
to \([-1,1]\).”

##### v0.4 — Balanced ternary, places \(3^i\)

**Alphabet:** digits \(d_i \in \{-1, 0, +1\}\) (**trits**), length \(n\) (default **10**).

**Place values base 3:** digit \(i\) is worth **\(3^i\)** (1, 3, 9, 27, …).

**Coding (balanced ternary):** form

\[
s = \sum_i d_i \cdot 3^i
\]

which uniquely represents integers in a symmetric range around zero
\(\bigl[-(3^n-1)/2,\, (3^n-1)/2\bigr]\). Then normalize \(w = s / s_{\max}\) into about
\([-1, 1]\).

**Why “balanced”:** digits can be negative, zero, or positive, so you do not need a
separate sign bit in the same way as ordinary binary; zero digits give **natural
sparsity**.

**Update** (separate column): carry-safe steps on that integer, then re-encode to
ternary digits.

**Intuition:** “Each weight is a **base-3 number** with digits −1, 0, +1, mapped to a
real weight in \([-1,1]\).”

##### Side-by-side (coding only)

| Version | What is stored per weight | How pieces are valued | Forward weight |
|---------|---------------------------|------------------------|----------------|
| **v0.1** | \(S\) values in {−1,+1} | **Equal** (unary) | Mostly **±1** via majority (coarse) |
| **v0.2** | \(n\) values with place \(2^i\) | **Powers of 2** | **Many levels** via \(s_{\mathrm{norm}}\) |
| **v0.3** | \(n\) bits in {0,1} | **Powers of 2** as integer \(v\) | **Many levels** via scaled \(v\) |
| **v0.4** | \(n\) digits in {−1,0,+1} | **Powers of 3** | **Many levels** + possible **zeros** |

### 2.2 Design decisions locked in

| Decision | Choice |
|----------|--------|
| Dataset for early representation science | **MNIST first**; scale task later when laws appear |
| Agent storage | Discrete buffers (semantic int; not FP master \(W\)) |
| Grad path | Float view for pressure; discrete state for storage/update |
| Naming | `experiments/v0_N/`, `results/v0_N/`, `checkpoints/v0_N/` |

### 2.3 Infrastructure (also done)

| Item | Location |
|------|----------|
| Dataset download (local only; not CI) | `scripts/download_datasets.py` |
| Gitignore for data / checkpoints / results | `.gitignore` |
| DuckDB experiment store | `binary_optimizers/store/` |
| STE vs Swarm harness (optional later) | `experiments/ste_vs_swarm/` |
| Unit tests | `experiments/v0_*/test_*.py`, `tests/test_store.py` |

### 2.4 Open gaps (re-prioritized)

| Gap | Priority now |
|-----|----------------|
| **Laws of swarm / register width** (8 … 1024+) | **Primary** |
| **Laws of exponential / float-like encoding** (exponent vs mantissa) | **Primary** |
| Harder data (CIFAR, small LM) to stress those laws | After laws are sketched on MNIST |
| STE head-to-head, multi-seed, hyperparameter polish | **Deferred** (not the research question) |
| Bit packing for production memory claims | Later engineering |
| CI without datasets | Keep enforcing |

---

## 3. Architecture of the current reference stack (plain language)

**Reference implementation: v0.3** (carry-safe binary place-value).  
v0.4 = same network with ternary digits. Use these as **scaffolds** to vary width and encoding—not as final optima.

### 3.1 Network shape (fixed while studying representation)

```
Image 28×28 → Flatten (784)
  → [optional LayerNorm]
  → DiscreteLinear: 784 → 128
  → ReLU
  → [optional LayerNorm]
  → DiscreteLinear: 128 → 10
```

BitNet-inspired in spirit (low-bit linears, optional pre-norm). Not a large LLM.

### 3.2 Weight as a discrete register (v0.3)

Each **weight** stores **n** bits \(b_i \in \{0,1\}\):

1. Integer \(v = \sum_i b_i 2^i\) in \(\{0,\ldots,2^n-1\}\).  
2. Weight \(w = 2v/(2^n-1) - 1 \in [-1,1]\).

No separate FP master \(W\). Carry-safe update: change \(v\) by ±Δ, re-encode bits
(carries exact). See older §3 writeup for full step-by-step if needed.

### 3.3 Two different meanings of “swarm size”

Do not confuse these—they answer different research questions:

| Concept | Where it appeared | What the number means |
|---------|-------------------|------------------------|
| **Unary population size** \(S\) | v0.1 | How many equal-weight ±1 agents vote for one logical weight |
| **Register width** \(n\) (bits / trits) | v0.2–v0.4 | How many **place-value digits** encode one weight |

Both are “how much discrete state **per weight**.”  
**Question A below** covers both families and asks whether a shared law exists.

---

## 4. Core research program (big picture)

### 4.1 Question A — Swarm / register size: what does width buy?

**Plain question.** For each **weight** we store a **batch of discrete units**
(agents or bits). How large should that batch be—8, 16, 32, 64, 128, 1024, more?

**Why it might matter**

| Small width | Large width |
|-------------|-------------|
| Few distinct weight levels; coarse steps | Many levels; fine steps and large dynamic range |
| Cheap memory and updates | Expensive **per weight** |
| May underfit or oscillate | May over-parameterize the *representation* (not the network topology) |

This is **not** the same as “wider network” (more neurons → more weights). It is
**more bits of state per weight**, with the layer shape held fixed.

**Hypotheses (to test, not assume)**

1. **Fixed sweet spot** — e.g. “around 16–32 always works on MNIST-like nets.”  
2. **Saturation** — accuracy rises with width then flattens; extra bits waste.  
3. **Scaling-law style** — optimum width grows with model size, data size, or depth
   (analogous to “more data ↔ more parameters” in neural scaling):
   - larger nets / harder tasks may need **more** bits per weight for fine updates;  
   - or the opposite: large nets tolerate **coarser** weights (redundancy).  
4. **Family-dependent** — unary \(S\) and place-value \(n\) have different curves;
   only one family has a simple law.

**Experimental skeleton (representation science, not tuning)**

- Fix: architecture family, data, train recipe defaults (do not grid-search lr for each width).  
- Vary **only** width on a log grid: e.g. \(8, 16, 32, 64, 128, 256, 512, 1024\)
  (and optionally 4 if cheap).  
- Families at least:
  - **Unary Swarm** (v0.1-style majority / sum), width = \(S\);  
  - **Carry-safe binary register** (v0.3), width = \(n_{\mathrm{bits}}\);  
  - Optionally **ternary register** (v0.4), width = \(n_{\mathrm{trits}}\).  
- Metrics: final / best accuracy, train dynamics (does it plateau or stall),
  update activity (how often high vs low digits move), effective use of range
  (do weights sit at extremes only?).  
- **Scaling axis (phase 2 of the same question):** repeat 2–3 widths on a
  **larger** model and/or harder data (deeper MLP, CIFAR) and see if the
  preferred width **moves** systematically.

**What “optimum” would mean**

- Not “best hyperparameter on one plot,” but either:  
  - a **stable plateau** (“≥32 bits adds little on this task”), or  
  - a **rule** (“preferred \(n\) scales like \(\log(\#params)\) or with depth”),  
  - or evidence that **no universal constant** exists—only trade-offs.

**Success for Question A:** a clear picture of accuracy vs width for ≥2 representation
families, plus a first statement on whether the curve is task/model-dependent.

---

### 4.2 Question B — Exponential / float-like encoding: exponent vs mantissa

**Plain question.** Real floating-point numbers split bits into **sign**, **exponent**
(order of magnitude), and **mantissa** (fine detail). Our discrete weight should
imitate a useful range of magnitudes. **How should we split the discrete budget?**

Today v0.2–v0.3 mostly use **uniform place value** (pure binary integer / \(2^i\)),
which is like a fixed-point number, **not** a full float. v0.4 is base-3 fixed-point-like.
We have **not** systematically designed true **exponent + mantissa** swarms.

**Encoding families to compare (big-picture methods, not micro-tweaks)**

| Encoding | Idea |
|----------|------|
| **Fixed-point / pure place-value** | All digits are mantissa-like; \(w \propto \sum d_i \beta^i\) (current v0.3/v0.4) |
| **Biased exponent + mantissa** | Some digits choose a scale \(2^e\) or \(3^e\); others fill significand |
| **Shared block scale** | One scale per row/block (microscaling-style), digits only for local shape |
| **Log-domain / multiplicative** | Digits update multiplicative factors (closer to LNS); different update rule |
| **Unary vs place-value vs float-split** | Same total bit budget, different structure |

**Sub-question: which bits are exponent, which are mantissa?**

For a fixed budget \(n\) bits, try allocations such as:

| Exponent bits \(n_e\) | Mantissa bits \(n_m\) | Intuition |
|----------------------|----------------------|-----------|
| 0 | \(n\) | Pure fixed-point (status quo v0.3) |
| 2–4 | \(n - n_e\) | Small dynamic range control |
| \(n/2\) | \(n/2\) | Balanced split |
| large \(n_e\), tiny \(n_m\) | Coarse scales, little fine detail |

Example decode (conceptual, one design among many):

\[
w = (-1)^{s}\,\cdot\, 2^{e - e_{\mathrm{bias}}}\,\cdot\,(1 + m/M)
\]

with \(s,e,m\) packed from the swarm digits. **Many designs are valid**—the research
is which **class** works for learning under discrete carry-safe (or Swarm) updates,
not which lr wins.

**Hypotheses**

1. **Fixed-point is enough** for MNIST-scale nets; float split only helps harder tasks.  
2. **A few exponent bits** unlock training (avoid vanishing small updates / stuck scales)
   without needing huge total \(n\).  
3. **Optimum split moves** with width \(n\) (e.g. exponent grows slowly, like
   \(\lceil\log_2 \log_2 n\rceil\) or stays constant).  
4. **Block scales** beat per-weight exponents (cheaper, closer to real low-bit LLM practice).

**Experimental skeleton**

- Fix total budget \(n \in \{16, 32, 64\}\) first (link to Question A).  
- For each \(n\), sweep **structures** (not tiny hyperparams): pure fixed-point vs
  several exponent/mantissa splits vs one block-scale baseline.  
- Same train defaults; measure accuracy **and** whether weights use many scales
  (histogram of exponents / magnitudes).  
- Later: same encodings on larger models / CIFAR to see if the best structure **moves**.

**Success for Question B:** a ranking of **encoding classes**, a recommended default
split rule (or “depends on \(n\) as follows…”), and evidence whether float-like
structure is necessary for the Swarm idea or optional.

---

### 4.3 How A and B interact (the real “scaling” picture)

| | Narrow register | Wide register |
|--|-----------------|---------------|
| **Fixed-point** | Coarse grid | Fine grid, limited dynamic range unless \(n\) is large |
| **Exponent + mantissa** | Few scales, coarse detail | Both range and detail—but need a good split |

Research products we want (analogous to scaling-law *statements*, not just tables):

1. **Width law:** accuracy vs \(S\) or \(n\); plateau or dependence on model/data size.  
2. **Structure law:** given \(n\), how to allocate exponent vs mantissa (or “use fixed-point”).  
3. **Joint rule (stretch goal):** e.g. “total bits grow slowly with depth; exponent bits stay ~3–5.”

Until those exist, “optimum is 16” or “optimum is 1024” would be **guesses** for one toy net.

---

## 5. Proposed work packages (ordered)

### WP1 — Width atlas (Question A)  ← start here

| ID | Content |
|----|---------|
| `v0_5_width_unary` | Unary Swarm, \(S \in \{8,16,32,64,128,256,512,1024\}\) (as far as compute allows) |
| `v0_5_width_register` | v0.3 carry-safe, \(n_{\mathrm{bits}}\) on the same grid |
| Optional | v0.4 width grid for ternary |

**Protocol spirit:** one network size, one data regime, default train settings; vary width only.  
**Deliverable:** curves + short note: plateau? peak? blow-up at large \(S\)?

### WP2 — Encoding atlas (Question B)

| ID | Content |
|----|---------|
| `v0_6_encoding` | Fixed total \(n\); compare fixed-point vs exp/mantissa splits vs block scale |

**Deliverable:** which encoding class works; default split rule for the Swarm idea.

### WP3 — Do the optima move? (scaling-style)

Repeat a **sparse** subset of WP1/WP2 on:

- deeper / wider MLP, and/or  
- CIFAR-scale vision toy  

Ask: does preferred width or preferred exp/mant split **shift** with scale?

### Explicitly deferred

- Multi-seed statistics for tiny gaps  
- STE vs Swarm scoreboard optimization  
- Hyperparameter grids (lr, momentum, recruit_rate) except when a representation is **broken** without a minimal fix  
- Full LLM pretrain, pure integer backward, committing dataset blobs  

---

## 6. Suggested experiment IDs

| ID | Focus |
|----|--------|
| `v0_5_width_unary` | Question A, unary family |
| `v0_5_width_register` | Question A, carry-safe binary |
| `v0_6_encoding` | Question B, exp vs mantissa structures |
| `v0_7_scale_shift` | Do optima move with model/data size? |

PROTOCOLS under `experiments/<id>/PROTOCOL.md`; log to DuckDB when useful.

---

## 7. Recommended immediate action

1. Keep roadmap on **`plan/swarm-roadmap`** (this document).  
2. Implement **`v0_5_width_register`** first (v0.3 scaffold, width grid including large \(n\)
   as far as memory allows)—answers “is 16 special or just what we tried?”  
3. Parallel or next: **`v0_5_width_unary`** so unary \(S\) and register \(n\) are not mixed.  
4. Then **`v0_6_encoding`** with fixed total bit budget and several exp/mantissa designs.  
5. Only after curves exist: sparse scaling check (deeper net / CIFAR).

**Default scaffold:** v0.3 carry-safe for register experiments; v0.1-style for pure swarm population size; v0.4 when ternary structure is the object of study.

---

## 8. How to re-run existing references

```bash
uv sync --extra bench
python scripts/download_datasets.py --mnist --no-cifar

python experiments/v0_3/train.py --ln-mode none   # reference register
python experiments/v0_1/train.py --ln-mode none   # reference unary (if CLI matches)
```

Unit tests / CI: `pytest experiments/v0_*/test_*.py tests/` — **no** dataset download.
