# Swarm / latent-free training — roadmap

**Branch:** `plan/swarm-roadmap`  
**Last updated:** 2026-07-31 (plain-language expansions for §2.4, §3, next-step #3)  
**Status:** v0.1–v0.4 complete; tooling for comparison/store partially done

This document is the living plan for latent-free binary/ternary Swarm optimizers
(BitNet-inspired layers, no FP master weights). Experiment folders use semantic
minor versioning: `experiments/v0_N/`.

---

## 1. Goal (research claim)

Train networks whose **weight state is discrete** (binary agents / bits / trits),
updated by **Swarm rules** (votes, place-value registers, carry-safe ±Δ), so
training does **not** keep a full-precision latent \(W\) (unlike BitNet QAT / STE).

| Claim | In scope today |
|-------|----------------|
| No FP master weight for linear layers | Yes (v0.1–v0.4) |
| Discrete optimizer update (flips / integer Δ) | Yes |
| Bit-only backward / integer LayerNorm | **No** (autograd + LN math still FP) |
| Trillion-param scale | **No** (MNIST ablations first) |

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

**Takeaways**

1. **Carry-safe integer steps (v0.3)** are the large win over independent flips (v0.2).
2. **Ternary (v0.4)** matches v0.3 accuracy with natural ~30% zero sparsity.
3. LayerNorm modes: with place-value + carry-safe, LN helps; pure `none` is still strong.
4. Half the agent budget (16 bits vs 32 unary) is enough once place-value is used.

Numbers also logged in DuckDB `results/experiments.duckdb` (when store ingest ran).

### 2.2 Design decisions locked in

| Decision | Choice |
|----------|--------|
| Dataset for optimizer science | **MNIST only** (CIFAR later) |
| Agent storage | int8 buffers (semantic int; not `nn.Parameter` master \(W\)) |
| Grad path | Float view + decode; Swarm updates discrete state |
| LN ablations | `none` / `no_affine` / `affine` (affine = optional FP crutch) |
| Naming | `experiments/v0_N/`, `results/v0_N/`, `checkpoints/v0_N/` |

### 2.3 Infrastructure (also done)

| Item | Location |
|------|----------|
| Dataset download (local only; not CI) | `scripts/download_datasets.py` |
| Gitignore for `data/`, checkpoints, results | `.gitignore` |
| DuckDB experiment store | `binary_optimizers/store/` |
| Report scripts | `scripts/report.sh`, `scripts/report_ste_vs_swarm.sh` |
| STE vs Swarm shared protocol | `experiments/ste_vs_swarm/` |
| Unit tests | `experiments/v0_*/test_*.py`, `tests/test_store.py` |

### 2.4 Incomplete / weak spots (plain language)

This section lists **gaps**: unfinished experiments or results that are **too weak to trust**
for a scientific claim. It is not mainly “code is broken,” but “the story is incomplete.”

#### Full STE vs Swarm comparison table (incomplete)

**Goal.** Train several methods on the **same** small network, same MNIST data, and the
same LayerNorm options, then compare fairly:

| Method | Plain idea |
|--------|------------|
| **STE** | Keep floating-point weights; use only ±1 in the forward pass; update the float weights with SGD |
| **Swarm v0.3** | No float “master” weight; store bits; update with carry-safe integer steps |
| **Swarm v0.4** | Same idea with ternary values {-1, 0, +1} |

LayerNorm is tested three ways (`none` / `no_affine` / `affine`), so a full comparison is a
**grid** of method × LayerNorm mode.

**Reality today.** The results database only has **one** STE run: STE + no LayerNorm, with
test accuracy about **21%** (near chance for 10-class MNIST). That run almost certainly
**failed** (bad learning rate, stopped early, or broken setup). It is **not** a valid STE
baseline.

**Why it matters.** Without a working STE baseline under the same rules, we cannot honestly
say Swarm is better or worse than STE—only that Swarm variants improved among themselves
(v0.1–v0.4).

#### Multi-seed error bars (not run)

Almost all numbers used a **single random seed** (42). Training is noisy: another seed can
change accuracy by a few points. **Error bars** means train e.g. 3 seeds and report
mean ± spread. Single-seed results are fine for development, weak for claims like
“method A beats method B.”

#### `n_bits` / `n_trits` trade-off curve (not run)

We fixed **16 bits** (v0.3) and **10 ternary digits** (v0.4). We have **not** asked:
*if we use fewer bits, how much accuracy do we lose?* That efficiency trade-off is unfinished.
(See next-step **#3** below for the full explanation.)

#### CIFAR / Transformer (not started)

**MNIST** is a small, easy digit task—good for debugging optimizers. **CIFAR-10** (color
photos) and **Transformers** (deeper language-style models) are harder. We have not shown
Swarm still trains well beyond the tiny MNIST MLP.

#### Packing bits for memory claims (only prototype)

In research code, each agent is often stored as `int8` (8 bits of RAM) even though the value
is only 0/1. **Packing** means storing eight binary agents in one byte (closer to 1 bit each)—
needed for a serious “uses less memory than float training” claim. Accuracy treats weights as
discrete, but **memory measurements still look like a lab prototype**, not a packed system.

#### CI without large datasets (policy to keep enforcing)

Automated tests on every commit should **not** download MNIST/CIFAR. Unit tests use tiny
synthetic tensors; people download data locally with `scripts/download_datasets.py`. Keep CI
from accidentally running full training that needs real data.

| Item (short) | Status |
|--------------|--------|
| Full STE vs Swarm grid | Incomplete (only broken STE `ln_none` ~0.21) |
| Multi-seed error bars | Not run (seed=42 only) |
| bits/param vs accuracy curve | Not run |
| CIFAR / Transformer | Not started |
| Bit packing for memory claims | Prototype only |
| CI without datasets | Policy agreed; keep enforcing |

---

## 3. Architecture of the current “winner” (plain language)

**Winner for next baselines: experiment v0.3** (carry-safe binary place-value Swarm).  
v0.4 is the same network shape with **ternary** digits instead of binary bits.

### 3.1 What the network looks like

We use a small **multi-layer perceptron** on MNIST (not a full BitNet LLM). Data flow:

```
Input image (28×28)
  → Flatten to 784 numbers
  → [optional LayerNorm]
  → CarrySafeLinear: 784 → 128   (first fully connected layer)
  → ReLU
  → [optional LayerNorm]
  → CarrySafeLinear: 128 → 10    (scores for digits 0–9)
  → Cross-entropy loss
```

| Piece | Role in plain terms |
|-------|---------------------|
| **Flatten** | Turn the image into a vector of 784 pixel values |
| **LayerNorm (optional)** | Rescale activations so training is more stable; three modes: off, normalize only, or normalize + small learnable scale/shift |
| **CarrySafeLinear** | Our special linear layer: weights are **not** free floats; they are decoded from discrete bits |
| **ReLU** | Standard “keep positive values” nonlinearity between the two layers |
| **10 outputs** | One score per digit class |

This is **BitNet-inspired** only in spirit (low-bit linear layers, optional pre-layer norm, no bias on those linears). It is **not** Microsoft’s large language model.

### 3.2 How one weight is stored (the discrete “register”)

For each connection from an input unit to an output unit we store **n bits**
(default `n_bits = 16`), each bit \(b_i\) is only **0 or 1**.

1. Interpret the bits as a normal binary **integer**:
   \[
   v = b_0\cdot 2^0 + b_1\cdot 2^1 + \cdots + b_{n-1}\cdot 2^{n-1}
   \]
   so \(v\) is an integer from \(0\) to \(2^n - 1\) (for 16 bits: 0 … 65535).

2. Map that integer to a **weight** in roughly \([-1, +1]\):
   \[
   w = \frac{2v}{2^n - 1} - 1
   \]
   Small \(v\) → weight near \(-1\); large \(v\) → weight near \(+1\); middle → near \(0\).

So the “true” parameter is the **bit string** (or the integer \(v\)). The floating value \(w\)
is only a **decoded** number used in the matrix multiply. There is **no separate full-precision
master weight** that STE would keep updating forever.

In code, bits may sit in an `int8` buffer for convenience; **semantically** they are still only
0/1. Packing them tightly for memory is a later efficiency step (see §2.4).

### 3.3 How a forward pass uses those weights

For a layer \(y = W x\) (no bias):

1. Decode every entry of \(W\) from its bits → \(w\) as above.  
2. Multiply by a **fixed** scale \(1/\sqrt{\text{fan-in}}\) so signals do not explode
   (standard practice for binary/low-bit nets; this scale is **not** a learned float weight matrix).  
3. Compute the usual linear map with those \(w\) values.

Gradients for training are obtained by treating the decode as differentiable for a moment
(autograd on a float view of the bits). That is only to get a **direction and strength of
pressure** on each weight. The **stored** state is still updated discretely (next subsection).

### 3.4 How the optimizer updates (carry-safe Swarm)

Classic float training: \(w \leftarrow w - \text{learning rate}\times\text{gradient}\).

v0.3 instead:

1. From the batch, get a gradient signal for each decoded weight \(w\)
   (how much the loss wants \(w\) to go up or down).  
2. Smooth that signal over batches with an **exponential moving average** (EMA)—like a short
   memory so one noisy batch does not thrash the bits.  
3. With a probability that grows with the size of that pressure, change the **integer** \(v\):
   - If the loss wants a smaller \(w\), decrease \(v\).  
   - If it wants a larger \(w\), increase \(v\).  
   - The step size \(\Delta v\) is **adaptive** (small pressure → small step; large → up to a cap).  
4. **Write bits back from the new integer** (standard binary encoding).  
   Any “carry” (e.g. 0111 → 1000 when adding one) is handled automatically by integer arithmetic.
   That is what **carry-safe** means: we do **not** flip unrelated bits independently and hope
   the place values still make sense.

After every step, bits remain only in \(\{0,1\}\).

### 3.5 LayerNorm modes (the three ablations)

| Mode | Meaning |
|------|---------|
| `none` | No LayerNorm—purest discrete-weight story |
| `no_affine` | LayerNorm normalizes mean/variance only; **no** extra learnable float scale/shift |
| `affine` | Full LayerNorm with small learnable scale and shift, trained with ordinary SGD |

In the best MNIST runs, `affine` and `no_affine` scored highest, but `none` still did well (~96%).
The optional LN parameters are the only “normal” float parameters in the affine case; the
**linear weights** stay discrete.

### 3.6 What v0.4 changes (ternary twin)

Same network shape. Instead of bits in \(\{0,1\}\) with place values \(2^i\):

- Digits in \(\{-1, 0, +1\}\)  
- Place values \(3^i\) (balanced ternary)  
- Same idea: form an integer, decode to \(w \in [-1,1]\), update with carry-safe integer steps  

Many digits can be **zero**, so weights are naturally sparse (~30% zeros in our runs).
Accuracy on MNIST was essentially the same as v0.3.

### 3.7 Why this is the “winner”

Among v0.1–v0.4, **v0.3** gave the best balance of:

- high MNIST accuracy,  
- clear discrete storage,  
- updates that respect place value (unlike independent bit flips in v0.2),  
- a simple story for the next baseline comparisons (vs STE, vs bit budget sweeps).

**Default for new experiments:** use **v0.3** unless the question is specifically about ternary
(v0.4) or about older ablations (v0.1/v0.2).

---

## 4. Proposed next steps

Ordered for **learning rate of research**, not paper length.

### P0 — Close the comparison (short)

1. **Finish `ste_vs_swarm`** under the shared protocol  
   - Run `ste_sgd` × all LN modes with working hyperparameters (current `ln_none` ~20% is not a valid STE baseline).  
   - Run `swarm_v0_3` / `swarm_v0_4` through the same harness into DuckDB.  
   - Produce one report table: method × LN × best test × wall time.  
2. **Multi-seed (3 seeds)** for v0.3 `none` + `affine` and STE at best LN — error bars for claims.

**Exit:** one markdown table proving Swarm vs STE under matched architecture.

### P1 — Efficiency science (medium)

#### 3. Pareto: bits per parameter vs accuracy (plain language)

**Question.** If we store **fewer bits for each weight**, how good does the model stay?

Today we only know that **16 bits** (v0.3) and **10 ternary digits** (v0.4) work well. We do
**not** know the cheapest setting that still works.

**What “bits per parameter” means.** A *parameter* here is one connection (one entry of a weight
matrix). Spending 16 bits per connection is twice as expensive as 8 bits per connection, even if
the network has the same number of connections. Plotting **bits per weight** (not only total
model size) isolates coding cost from “we just made the network wider.”

**What “Pareto” means here.** A simple trade-off plot:

- **X-axis:** cost — bits spent per weight (or total agent bits)  
- **Y-axis:** quality — test accuracy on MNIST  

Train several settings and plot one point each:

| Family | Settings to try |
|--------|-----------------|
| v0.3 | `n_bits` ∈ {4, 8, 12, 16, 24} |
| v0.4 | `n_trits` ∈ {4, 6, 8, 10, 12} |

Connecting the points shows the **frontier**: for a given bit budget, what accuracy you get.
Optionally mark v0.1 (32 unary agents per weight) on the same figure for comparison.

**What we would learn**

- Is 8 bits enough for high MNIST accuracy, or do we need 16?  
- Does ternary with fewer digits beat binary at the **same bit budget**?  
- Does the method still look attractive if the competitor is “8-bit Adam + float weights,” not only raw accuracy?

**How to run (conceptually)**

1. Fix network size, data, and (ideally) several seeds.  
2. Vary **only** `n_bits` or `n_trits`.  
3. Train to early stopping.  
4. Plot accuracy vs bits per weight.

#### 4. Memory accounting

- Report true storage if packed (1 bit per binary agent; ternary is denser but not a power of two)
  vs the current `int8` research prototype.  
- Optional: pack/unpack in checkpoints without changing training math.

**Exit (P1):** a curve showing where carry-safe Swarm wins or loses on a bits-per-parameter story
(not only on accuracy).

### P2 — Harder tasks (medium–long)

5. **CIFAR-10** small BitNet-style CNN or MLP-mixer toy with **v0.3** (and STE control).  
6. **Tiny LM** (e.g. nanoGPT-scale BitLinear + carry-safe Swarm on char/token LM) — only after CIFAR is stable.

**Exit:** not SOTA; prove training doesn’t die at depth / non-MNIST.

### P3 — Algorithm polish (parallel tracks)

7. **v0.5 candidates** (pick one primary):  
   - **A.** Hybrid: carry-safe register + population voting only on residual/error.  
   - **B.** Hardware-friendly: bit-serial update kernels / popcount metrics.  
   - **C.** Stronger STE baseline (Adam on latent BitLinear, same topology) for fairer bar.  
8. Theory note: expected Δ vs SGD step; when carry-safe ≈ quantized SGD.

### Explicitly deferred

- Full BitNet 2B / LLM pretrain.  
- Pure integer backward.  
- Integer LayerNorm redesign.  
- Committing MNIST/CIFAR blobs (keep `scripts/download_datasets.py`).

---

## 5. Suggested experiment IDs

| ID | Focus |
|----|--------|
| `ste_vs_swarm` (finish) | P0 comparison |
| `v0_3_bitsweep` | P1 n_bits Pareto |
| `v0_4_tritsweep` | P1 n_trits Pareto |
| `v0_5_cifar` or `v0_5_*` | P2 / P3 chosen track |

Keep PROTOCOLS under `experiments/<id>/PROTOCOL.md` and log to DuckDB when using the store.

---

## 6. Recommended immediate action

1. Stay on (or merge) roadmap docs from branch **`plan/swarm-roadmap`**.  
2. Open implementation branch **`feat/ste-vs-swarm-complete`** (or continue `experiments/ste_vs_swarm`):  
   - Fix STE hyperparameters until MNIST STE ≥ ~97% under shared MLP+LN protocol (or document ceiling).  
   - Fill full method × LN matrix into DuckDB.  
3. Then **`feat/v0_3-bitsweep`** for the efficiency plot.

**Default “product” of the ladder today:** treat **v0.3 carry-safe** as the binary Swarm baseline; **v0.4** as the ternary twin.

---

## 7. How to re-run

```bash
# Local data (not for CI)
uv sync --extra bench
python scripts/download_datasets.py --mnist --no-cifar

# Ladder experiments
python experiments/v0_3/train.py --ln-mode all
python experiments/v0_4/train.py --ln-mode all

# Comparison (when harness is fully green)
python experiments/ste_vs_swarm/train.py
./scripts/report_ste_vs_swarm.sh
```

Unit tests / CI: `pytest experiments/v0_*/test_*.py tests/` — **no** dataset download.
