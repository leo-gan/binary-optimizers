# Training memory at 1B parameters

**Purpose:** static (parameter + optimizer + gradient) memory for **1 billion**
scale under regimes that matter to this project.

**Scope:** weight storage, optimizer state, and resident gradients.  
**Out of scope:** activations, KV cache, dataloader, CUDA workspace, and
communication buffers. Those often dominate wall-clock peak RAM at LLM scale;
this page isolates the **weight-path** story the Swarm design targets.

**Units:** tables use **GB** as \(10^9\) bytes-scale decimal
(\(N \times\) bytes-per-unit / \(10^9\)). Multiply by \(1.074\) for GiB.

### Counting convention (read first)

| Regime | What “1B” means here | Unit |
|--------|----------------------|------|
| All FP, STE, **register** Swarm | \(N_{\mathrm{link}} = 10^9\) matrix entries | One **link** ≈ one classical parameter |
| **Unary link Swarm** (this design) | \(N_{\mathrm{w}} = 10^9\) **weights** | One **weight** = **one bit** in a swarm ([terminology](UNARY_SWARM_TERMINOLOGY.md)) |

**Yes — for Unary Swarm, one swarm bit = one weight is the correct interpretation**
under the frozen glossary. Optimizer state (\(g, m, v\)) still attaches **per link**,
not per bit, so:

\[
N_{\mathrm{link}} = N_{\mathrm{w}} / S,\qquad
\text{swarm bits} = N_{\mathrm{w}} = S \cdot N_{\mathrm{link}}.
\]

The old mistake of “1B links × \(S{=}256\) bits” inflates memory by \(S\) and is
**not** used for Unary rows below.

---

## 1. Summary table

### 1.1 Classical 1B **links** (FP / STE / register Swarm)

| Case | Weight path | Optimizer path | Bytes / link | **Static @ \(10^9\) links** | vs FP32 Adam |
|------|-------------|----------------|-------------:|----------------------------:|-------------:|
| **All FP** (FP32 + Adam) | FP32 \(W\) | \(m, v\) FP32 | **16** | **16.0 GB** | 1.0× |
| **Binary NN + STE** (SGD) | FP32 master; `sign` forward | none | **8** | **8.0 GB** | 0.50× |
| **Binary NN + STE + Adam** | FP32 master | \(m, v\) FP32 | **16** | **16.0 GB** | 1.0× |
| **Binary register Swarm** (\(n{=}8\), packed) | 8 bits/link | FP32 pressure EMA | **5** | **5.0 GB** | 0.31× |
| **Ternary register Swarm** (\(n{=}8\), packed) | 16 bits/link (2 b/trit) | FP32 pressure EMA | **6** | **6.0 GB** | 0.38× |

### 1.2 Unary link Swarm — 1B **weights (bits)**

Packed bits; \(g,m,v\) FP32 **per link**; \(S\) = swarm size.

**Bytes per weight (bit):**

\[
b_{\mathrm{SGD}} = \frac{1}{8} + \frac{4}{S},\qquad
b_{\mathrm{Adam}} = \frac{1}{8} + \frac{12}{S}
\]

(1/8 B for the packed bit; 4 B of \(g\) amortized over \(S\) bits; Adam adds \(m,v\) → 12 B / link / \(S\).)

| Case | \(S\) | Bytes / weight (bit) | **Static @ \(10^9\) bits** | vs FP32 Adam @ 1B **links** |
|------|------:|---------------------:|---------------------------:|----------------------------:|
| **Unary + SGD** | 128 | \(0.125+0.03125=0.156\) | **0.156 GB** | ~100× smaller (different unit) |
| **Unary + SGD** | **256** | \(0.125+0.0156=0.141\) | **0.141 GB** | — |
| **Unary + Adam** | 128 | \(0.125+0.0938=0.219\) | **0.219 GB** | — |
| **Unary + Adam** | **256** | \(0.125+0.0469=0.172\) | **0.172 GB** | — |

**Fair “same network topology” comparison:** if the FP net has \(N_{\mathrm{link}}\) links,
Unary uses \(N_{\mathrm{w}} = S \cdot N_{\mathrm{link}}\) bit-weights. At \(S{=}256\) and
\(N_{\mathrm{link}}=10^9\):

| Case | Formula | Total |
|------|---------|------:|
| FP32 + Adam | \(10^9 \times 16\) | **16.0 GB** |
| Unary + SGD | \(10^9 \times (S/8 + 4) = 10^9 \times 36\) | **36.0 GB** |
| Unary + Adam | \(10^9 \times (S/8 + 12) = 10^9 \times 44\) | **44.0 GB** |

So: **per bit** Unary is tiny (~0.14–0.17 GB at 1B bits); **per link** at fixed topology
and \(S{=}256\) it is **heavier** than FP32 Adam because each link stores 256 bits + FP
optimizer on the link value. Do not mix the two headlines.

**Headline (Unary terms):** with **weight = bit**, 1B-weight Unary training static state
is **≪ 1 GB** at \(S \ge 32\) (SGD or Adam), because bits pack 8:1 and \(g/m/v\) amortize
across the swarm.

---

## 2. Assumptions (read before quoting numbers)

### 2.1 What counts as “static”

| Buffer | Counted? | Notes |
|--------|----------|--------|
| Stored weights / population | **Yes** | Persistent across steps |
| Optimizer state | **Yes** | Adam moments, pressure EMA, … |
| Gradients for weights | **Yes** | Resident `.grad` (or equivalent) at step time |
| Transient STE float view of population | **Noted separately** | Peak during autograd; can be large if materialised as FP32 `[…, n]` |
| Activations / KV | **No** | Task- and batch-dependent |

### 2.2 Defaults taken from this repo’s research ladder

| Symbol | Default used here | Source |
|--------|-------------------|--------|
| Register width \(n_{\mathrm{bits}}\) | **8** | WP1 peak (`v0_5_width_register`, `docs/OPTIMA_STATUS.md`) |
| Ternary width \(n_{\mathrm{trits}}\) | **8** | Same budget as binary peak for fair comparison (v0.4 default was 10; see §5) |
| Unary swarm size \(S\) | **256** (default); also **128** (U3 peak under sgd) | [UNARY_SWARM_TERMINOLOGY.md](UNARY_SWARM_TERMINOLOGY.md), ladder notes |
| Unary optimizer | **SGD** or **Adam per link** on link value | `UnaryLinkOptimizer` (`v0_8`); U2 prefers Adam |
| Register Swarm optimizer | **One FP32 pressure EMA per link** | `SwarmOptimizerV0{1,3,4}` (legacy unary flip path) |
| STE optimizer | **SGD + clamp** (no momentum) | `docs/optimizers.md` |
| Packing | **Ideal packed** bits for production claims | Research code stores **int8** swarm today (§6) |

### 2.3 “All FP” definition

**Primary baseline (table above):** full **FP32** weights + **Adam** (\(m\) and \(v\)
both FP32). This is the classic 16 bytes/param teaching model.

**Optional LLM-style mixed precision** (same total order of magnitude):

| Component | Bytes/param |
|-----------|------------:|
| Weight FP16 | 2 |
| Master FP32 (common) | 4 |
| Grad FP16 | 2 |
| Adam \(m, v\) FP32 | 4 + 4 |
| **Total** | **16** |

So mixed-precision AdamW with a master copy is also **~16 GB @ 1B**, not half of FP32.

### 2.4 Float formats: IEEE FP16 vs BF16 vs project “exp/mant” vs ExpFP

Do **not** conflate these four ideas. Only the first two are common hardware types
for \(W, g, m, v\); the last two are project / research designs.

| Name | Layout (typical) | Dynamic range | Role in *this* repo |
|------|------------------|---------------|---------------------|
| **IEEE FP16** (half) | 1 sign + **5** exp + 10 mant | **Narrow** (max ~6.5×10⁴) | Classic mixed precision; needs **loss scaling**; squared Adam \(v\) can **overflow / underflow** more easily than FP32 |
| **BF16** (bfloat16) | 1 sign + **8** exp + 7 mant | **Same exp field as FP32** | The usual “custom 16-bit float that keeps FP32’s exponent.” Wider range than FP16; **less** overflow risk; **coarser** precision (fewer mantissa bits). Often written **BF16** / bfloat16 — **not** “BP16” |
| **WP2 exp/mant coding** | Discrete register digits split into scale vs fine bits | Set by digit budget \(n\), not IEEE | **Weight encoding** experiment (`v0_6_encoding`). Not IEEE floats. Sketch result: **fixed-point beat** exp/mant on MNIST |
| **ExpFP / block-FP** (research notes) | Shared exponent per block + narrow mantissas | Rescaled per block | Proposed low-bit **grad / state** layout in `docs/temp/`; **not** the default Swarm pressure path today |

**Why overflow talk targeted IEEE FP16, not “all 16-bit floats”:**  
overflow/underflow of Adam’s second moment \(v \propto g^2\) is mainly an **IEEE FP16
range** problem (5-bit exponent). **BF16** keeps FP32’s 8-bit exponent, so large
magnitudes behave much more like FP32; the trade-off is **mantissa precision**, not
range. Many LLM stacks prefer **BF16 compute** for that reason, and still often keep
Adam \(m, v\) in **FP32** (or carefully quantized states) for accuracy of the
optimizer trajectory.

**What Swarm uses today for pressure EMA:** ordinary **FP32**, not BF16, not WP2
weight exp/mant, not ExpFP. Memory rows that say “4 B EMA” mean IEEE FP32 unless a
future packing note says otherwise.

**Rough static bytes if one *did* store \(g, m, v\) in 16-bit types** (still not our
default claim):

| Setup | Bytes/param (sketch) | @ 1B |
|-------|---------------------:|-----:|
| FP32 \(W,g,m,v\) (table baseline) | 16 | 16 GB |
| BF16 \(W,g\) + FP32 \(m,v\) (common-ish) | \(2+2+4+4 = 12\) | 12 GB |
| BF16 everything including \(m,v\) (aggressive) | 8 | 8 GB |
| IEEE FP16 everything including \(m,v\) (risky) | 8 | 8 GB |

Even under aggressive 16-bit Adam state, packed binary Swarm \(n{=}8\) (~5 B with FP32
EMA) remains competitive; the Swarm win is still **no master \(W\) + no Adam pair**.

---

## 3. Per-case breakdown

### 3.1 All FP — FP32 weights + Adam

| Component | Precision | Bytes / weight |
|-----------|-----------|---------------:|
| Weight \(W\) | FP32 | 4 |
| Gradient \(\partial L/\partial W\) | FP32 | 4 |
| Adam first moment \(m\) | FP32 | 4 |
| Adam second moment \(v\) | FP32 | 4 |
| **Total** | | **16** |

\[
M = N \times 16 = 10^9 \times 16~\mathrm{B} = \mathbf{16.0~\mathrm{GB}}
\]

No discrete network; no Swarm. This is the reference “train in floating point.”

---

### 3.2 Binary NN + STE optimizer

**Network claim:** forward uses binary weights \(\tilde w = \mathrm{sign}(W)\).  
**Storage reality:** the learnable state is still a **latent FP master** \(W\)
(BitNet / STE pattern; see `experiments/ste_vs_swarm`, `BitLinearSTE`).

#### A) Project STE — SGD on master (no Adam state)

| Component | Precision | Bytes / weight |
|-----------|-----------|---------------:|
| Master \(W\) | FP32 | 4 |
| Gradient | FP32 | 4 |
| Binary view \(\mathrm{sign}(W)\) | ephemeral | 0 (not stored) |
| Optimizer state | — | 0 |
| **Total** | | **8** |

\[
M = N \times 8 = \mathbf{8.0~\mathrm{GB}}
\]

**vs all-FP Adam:** 50% static memory. The binary *forward* does not free the master;
SGD (vs Adam) is what halves the budget.

#### B) Industry BitNet-style QAT — STE + Adam on master

| Component | Precision | Bytes / weight |
|-----------|-----------|---------------:|
| Master \(W\) | FP32 | 4 |
| Gradient | FP32 | 4 |
| Adam \(m, v\) | FP32 | 8 |
| **Total** | | **16** |

\[
M = \mathbf{16.0~\mathrm{GB}}
\]

**Same static footprint as all-FP Adam.** Quantizing the forward pass alone does
not solve the training-memory problem if master + Adam remain FP32.

---

### 3.3 Binary NN + Swarm optimizer (latent-free)

**Network:** discrete binary state only (no FP master \(W\)).  
**Optimizer:** discrete flips / carry-safe ±Δ on the register; **FP pressure EMA**
for inertia (BOP-like), not Adam moments on \(W\).

#### Primary design point — carry-safe register, \(n = 8\) (WP1 peak)

| Component | Representation | Bytes / weight |
|-----------|----------------|---------------:|
| Population (bits) | 8 × 1 bit, **packed** | 1 |
| Pressure EMA | FP32, one scalar per logical weight | 4 |
| Grad for step | FP32 on aggregated pressure path\* | 0–4 |
| **Total (persistent + typical grad)** | | **5** (EMA+bits; +0 if grad freed; **9** if full FP32 grad held) |

\*After backward, Swarm aggregates per-digit grads to one pressure scalar
(`grad_w` in `SwarmOptimizerV03`). Persistent state is **bits + EMA**. We count
**5 B/weight** when the resident gradient is that of the **logical weight** (or is
released after the step). If an implementation keeps a full FP32 `.grad` on a
materialised population, add \(4n\) temporarily (§3.3 peak note).

\[
M_{\mathrm{static}} \approx N \times 5 = \mathbf{5.0~\mathrm{GB}}
\]

**vs all-FP Adam:** ~0.31× (≈ **3.2× smaller**).  
**vs STE+SGD:** ~0.63× (≈ **1.6× smaller**).

#### Peak / implementation note (STE float view)

Research layers build a float view of the population for manual STE
(`[out, in, n]` FP32). That view is **not** the master weight, but during
forward+backward it can cost:

\[
n \times 4~\mathrm{bytes/weight} \quad (n{=}8 \Rightarrow 32~\mathrm{B/weight}~\mathrm{transient}).
\]

A production packing path should avoid retaining a full FP32 population tensor
(bit-packed storage + narrow STE or integer backward). Static claims above use
**packed persistent** state; quote peak RAM only with an explicit STE-view model.

---

### 3.4 Ternary NN + Swarm ternary optimizer (latent-free)

**Network:** balanced ternary digits \(d_i \in \{-1,0,+1\}\) with places \(3^i\)
(v0.4 lineage).  
**Optimizer:** carry-safe ±Δ on the ternary integer, then re-encode trits
(`SwarmOptimizerV04`); same **FP32 pressure EMA** pattern as binary Swarm.

#### Primary design point — \(n_{\mathrm{trits}} = 8\), 2 bits per trit packed

| Component | Representation | Bytes / weight |
|-----------|----------------|---------------:|
| Trit digits | 8 × 2 bits | 2 |
| Pressure EMA | FP32 | 4 |
| **Total** | | **6** |

\[
M \approx N \times 6 = \mathbf{6.0~\mathrm{GB}}
\]

**vs all-FP Adam:** 0.38× (≈ **2.7× smaller**).  
**vs binary Swarm \(n{=}8\):** +1 B/weight (ternary alphabet needs ~2 bits/digit).

#### Variant — v0.4 default width \(n_{\mathrm{trits}} = 10\)

| Component | Bytes / weight |
|-----------|---------------:|
| 10 trits × 2 bits | 2.5 |
| Pressure EMA | 4 |
| **Total** | **6.5** → **6.5 GB @ 1B** |

#### Information-theoretic floor (not used in totals)

A single ternary digit is \(\log_2 3 \approx 1.58\) bits. Packing to 1.58 b/trit is
awkward in practice; **2 bits/trit** is the engineering assumption above.
For \(n{=}8\): \(8 \times 1.58 \approx 12.6\) bits ≈ 1.58 B + 4 B EMA ≈ **5.6 B/weight**
if ideal coding were free.

---

### 3.5 Unary link Swarm (weight = bit) — SGD vs Adam

Terms: [UNARY_SWARM_TERMINOLOGY.md](UNARY_SWARM_TERMINOLOGY.md).  
Implementation: `experiments/v0_8_unary_link/` (`UnaryLinkOptimizer`).

#### Storage model

| Object | Count | Storage (production packing) |
|--------|------:|------------------------------|
| **Weight** | \(N_{\mathrm{w}}\) | 1 bit each → \(N_{\mathrm{w}}/8\) bytes |
| **Link** | \(N_{\mathrm{w}}/S\) | owns one swarm of \(S\) weights |
| **Link value** \(w_{\mathrm{link}}\) | \(N_{\mathrm{w}}/S\) | not stored; re-encoded from swarm |
| Grad \(g = \partial L/\partial w_{\mathrm{link}}\) | \(N_{\mathrm{w}}/S\) | FP32 → 4 B / link |
| Momentum \(m\) (SGD-M or Adam) | \(N_{\mathrm{w}}/S\) | FP32 → 4 B / link |
| Second moment \(v\) (Adam only) | \(N_{\mathrm{w}}/S\) | FP32 → 4 B / link |

No FP master weight matrix. Continuous optimizer lives on **link values** only.

#### Totals at \(N_{\mathrm{w}} = 10^9\) bit-weights

| Optimizer | Formula (bytes) | \(S{=}128\) | \(S{=}256\) | \(S{=}512\) |
|-----------|-----------------|------------:|------------:|------------:|
| **SGD** (only \(g\)) | \(N_{\mathrm{w}}/8 + 4 N_{\mathrm{w}}/S\) | **0.156 GB** | **0.141 GB** | **0.133 GB** |
| **SGD + momentum** | \(N_{\mathrm{w}}/8 + 8 N_{\mathrm{w}}/S\) | 0.188 GB | 0.156 GB | 0.141 GB |
| **Adam** (\(g,m,v\)) | \(N_{\mathrm{w}}/8 + 12 N_{\mathrm{w}}/S\) | **0.219 GB** | **0.172 GB** | **0.148 GB** |

Worked example **\(S{=}256\)** (plan default / U1 cell):

| Buffer | Count | Bytes |
|--------|------:|------:|
| Packed swarm | \(10^9\) bits | \(10^9/8 = 1.25\times 10^8\) (**0.125 GB**) |
| \(g\) | \(10^9/256\) links | \(4 \times 3.906\times 10^6\) (**0.0156 GB**) |
| \(m,v\) (Adam) | same | **0.0313 GB** |
| **SGD total** | | **0.141 GB** |
| **Adam total** | | **0.172 GB** |

Adam costs only **~0.031 GB** extra vs SGD at this scale (moments are few because
there are only \(N_{\mathrm{w}}/S\) links). Larger \(S\) → **fewer links** → **cheaper**
\(g,m,v\); swarm storage stays \(N_{\mathrm{w}}/8\) when bit count is fixed.

#### Same-topology view (1B **links**, each with swarm size \(S\))

Here \(N_{\mathrm{w}} = S \cdot 10^9\). Swarm alone is \(S/8\) GB; plus optimizer:

| \(S\) | Swarm | + SGD (\(g\)) | + Adam (\(g,m,v\)) |
|------:|------:|--------------:|-------------------:|
| 32 | 4.0 GB | **8.0 GB** | **16.0 GB** |
| 128 | 16.0 GB | **20.0 GB** | **28.0 GB** |
| 256 | 32.0 GB | **36.0 GB** | **44.0 GB** |
| 512 | 64.0 GB | **68.0 GB** | **76.0 GB** |

At fixed topology, big \(S\) is **memory-heavy**. That is the right table for
“replace every FP parameter by a swarm of size \(S\).”

#### Research code (int8, not packed)

Today each weight is **int8** → \(N_{\mathrm{w}}\) bytes for the swarm alone
(**1.0 GB** at 1B bits) before \(g,m,v\). Packed claims above are architectural.

---

## 4. Side-by-side

### 4.1 Classical: 1B **links**

```
All FP (FP32+Adam)          |################  16.0 GB
Binary STE + Adam           |################  16.0 GB
Binary STE + SGD            |########          8.0 GB
Unary S=256 + Adam (1B links)|############################################  44.0 GB
Unary S=256 + SGD  (1B links)|####################################  36.0 GB
Ternary register n=8        |######            6.0 GB
Binary register n=8         |#####             5.0 GB
```

### 4.2 Unary terms: 1B **weights (bits)**, \(S{=}256\)

```
Unary + Adam  |##  0.172 GB
Unary + SGD   |#   0.141 GB
(swarm alone) |#   0.125 GB
```

**Interpretation**

1. **Binary forward + STE + Adam** does not unlock 1B-**link** training memory; masters and
   moments still own the budget.  
2. **STE + SGD** helps (drops Adam), but still stores an FP master.  
3. **Latent-free Swarm** removes the master; static cost is **discrete width + one
   FP EMA**. At \(n{=}8\), that is the cheapest serious design point we have
   evidence for on MNIST-scale nets.  
4. **Ternary Swarm** is in the same ballpark as binary register Swarm (+~1 B/wt for
   2-bit trits), not in the 16 GB Adam regime.

---

## 5. Sensitivity: width and optimizer choices

### 5.1 Binary register width (packed bits + 4 B EMA)

| \(n_{\mathrm{bits}}\) | Bits | + EMA | B/wt | @ 1B | Note |
|----------------------:|-----:|------:|-----:|-----:|------|
| 4 | 0.5 | 4 | 4.5 | 4.5 GB | WP1 underfit side |
| **8** | **1** | **4** | **5** | **5.0 GB** | **WP1 peak** |
| 16 | 2 | 4 | 6 | 6.0 GB | Slightly below peak acc |
| 32 | 4 | 4 | 8 | 8.0 GB | WP1 cliff under fixed Δ recipe |

EMA (4 B) dominates once \(n\) is small; widening the register is cheap in RAM
until \(n \gtrsim 32\).

### 5.2 If pressure EMA is quantized later (stretch goal)

| Pressure state | Binary \(n{=}8\) B/wt | @ 1B |
|----------------|----------------------:|-----:|
| FP32 EMA (today) | 5 | 5.0 GB |
| FP16 EMA | 3 | 3.0 GB |
| 8-bit EMA | 2 | 2.0 GB |
| No EMA (threshold-only) | 1 | 1.0 GB |

Today’s ladder uses FP32 EMA; lower-bit pressure is **not** claimed as done.

### 5.3 STE with momentum SGD

Add 4 B/weight for a velocity buffer → binary STE+SGD-M ≈ **12 B** → **12 GB @ 1B**.

---

## 6. Research code vs production packing

| | Current `experiments/v0_*` storage | Packed claim (this doc) |
|--|------------------------------------|-------------------------|
| Binary agents / bits | `int8` buffer shape `[…, n]` or `[…, S]` | 1 bit per binary agent |
| Ternary digits | `int8` in \(\{-1,0,1\}\) | 2 bits per trit |
| Pressure EMA | FP32 tensor per layer | same |
| STE path | FP32 population view for autograd | should be narrowed / freed |

**Example:** binary \(n{=}8\) in research layout:

- Population: \(8 \times 1~\mathrm{B} = 8~\mathrm{B/weight}\)  
- EMA: \(4~\mathrm{B}\)  
- **Persistent ≈ 12 B/weight → 12 GB @ 1B** (before STE view)  
- Plus optional FP32 view: \(+32~\mathrm{B}\) peak during backward  

So **measured** PyTorch RSS on the toy ladder will **not** match the packed 5 GB
figure until bit-packing and STE-view elision land. The packed numbers are the
**architectural** budget for scaling arguments and hardware story; PLAN.md lists
bit packing as later engineering.

---

## 7. Worked totals (copy-paste)

### 7.1 \(N = 10^9\) **links** (classical parameter count)

| Case | Formula (bytes) | Total |
|------|-----------------|------:|
| All FP32 + Adam | \(N(4+4+4+4)\) | **16.0 GB** |
| Binary STE + SGD | \(N(4+4)\) | **8.0 GB** |
| Binary STE + Adam | \(N(4+4+4+4)\) | **16.0 GB** |
| Binary register Swarm \(n{=}8\) packed | \(N(1+4)\) | **5.0 GB** |
| Binary register Swarm \(n{=}8\) int8 | \(N(8+4)\) | **12.0 GB** |
| Ternary register Swarm \(n{=}8\) packed | \(N(2+4)\) | **6.0 GB** |
| Unary link Swarm \(S{=}256\) + SGD | \(N(S/8 + 4)=N\cdot 36\) | **36.0 GB** |
| Unary link Swarm \(S{=}256\) + Adam | \(N(S/8 + 12)=N\cdot 44\) | **44.0 GB** |

### 7.2 \(N_{\mathrm{w}} = 10^9\) **weights (bits)** — Unary only

| Case | Formula (bytes) | Total |
|------|-----------------|------:|
| Unary + SGD, any \(S\) | \(N_{\mathrm{w}}/8 + 4 N_{\mathrm{w}}/S\) | see §3.5 |
| Unary + Adam, \(S{=}256\) | \(10^9/8 + 12\cdot 10^9/256\) | **0.172 GB** |
| Unary + SGD, \(S{=}256\) | \(10^9/8 + 4\cdot 10^9/256\) | **0.141 GB** |
| Swarm bits only (packed) | \(N_{\mathrm{w}}/8\) | **0.125 GB** |

---

## 8. What not to claim

- **“Binary NN trains in 1/32 the memory”** — false if STE keeps FP32 masters (and
  worse if Adam is kept).  
- **Packed Swarm totals as current measured GPU use** — false until packing ships.  
- **Activation-free totals as full training RAM** — false for transformers; always
  state that this page is **weight-path static** memory.  
- **Mix bit-count and link-count** — “Unary is 0.14 GB at 1B” (bits) is compatible with
  “Unary is 36 GB at 1B links with \(S{=}256\)”; they answer different questions.  
- **Unary \(S{=}256\) is free on a fixed topology** — each link pays \(S\) bits.

---

## 9. Related docs

| Doc | Role |
|-----|------|
| [PLAN.md](PLAN.md) | Grand goal; WP1 width optima used here |
| [OPTIMA_STATUS.md](OPTIMA_STATUS.md) | \(n \approx 8\), \(S \approx 256\)–1024 |
| [UNARY_SWARM_TERMINOLOGY.md](UNARY_SWARM_TERMINOLOGY.md) | **Weight = bit**; link; link value |
| [UNARY_SWARM_EXPERIMENT_PLAN.md](UNARY_SWARM_EXPERIMENT_PLAN.md) | Unary ladder (v0_8+) |
| [SWARM_OPTIMIZER.md](SWARM_OPTIMIZER.md) | Latent-free vs STE master \(W\) |
| [optimizers.md](optimizers.md) | STE = SGD + clamp historical note |
| [TRAIN_BUDGET.md](TRAIN_BUDGET.md) | Wall-clock train protocol (orthogonal to RAM model) |

---

## 10. Changelog

| Date | Change |
|------|--------|
| 2026-08-05 | Initial 1B static memory comparison for FP / STE / binary Swarm / ternary Swarm |
| 2026-08-05 | §2.4: IEEE FP16 vs **BF16** (FP32 exp) vs WP2 exp/mant vs ExpFP |
| 2026-08-07 | Unary: weight=bit counting; SGD vs Adam @ 1B bits; dual tables (bits vs links) |
