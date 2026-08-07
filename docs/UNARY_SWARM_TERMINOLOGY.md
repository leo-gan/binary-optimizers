# Unary Swarm — terminology (frozen)

**Status:** locked design vocabulary for the **intended** Unary Swarm model  
**Date:** 2026-08-05  

This document freezes names and defaults for **Unary Swarm as specified below**.
It is **not** a description of the current `experiments/v0_1` implementation
(majority decode + pressure-EMA flips). See §5 for the map to legacy repo language.

---

## 1. Core terms

| Term | Definition |
|------|------------|
| **Weight** | One learnable **bit** of state, value in \(\{\pm 1\}\) (or \(0/1\) if packed). Atomic stored unit. |
| **Swarm** | A **batch of weights** (bits) treated as one group. |
| **Swarm size** \(S\) | Number of weights in that swarm (e.g. \(S = 256\)). |
| **Link** | One matrix entry \((i,j)\). **Owns exactly one swarm.** |
| **Link value** \(w_{\mathrm{link}}\) | Scalar **actually used in the matmul** for that link. |
| **Encoder** | Map from swarm → link value. |
| **Decoder** | Map from continuous update \(\Delta\) (and noise) → an **update swarm** of size \(S\). |
| **Update swarm** | Bit pattern of length \(S\) proposed for writeback (same size as the weight swarm). |
| **Merge op** \(\star\) | Bitwise rule that combines weight swarm with update swarm. |
| **Optimizer state** | Continuous buffers (\(g\), \(m\), \(v\), …) used by Adam/SGD. |
| **Minibatch** | Batch of **training examples** (data). Never use bare “batch” for a swarm. |

### 1.1 Link value (definition)

**Link value** \(w_{\mathrm{link}}\):

> The scalar number used in the linear map for one **link**, obtained by **encoding**
> that link’s **swarm**.

\[
w_{\mathrm{link}} = \mathrm{encoder}(\mathrm{swarm})
\]

- A **weight** is still one bit (stored).  
- The **swarm** is the full bit group on the link.  
- \(w_{\mathrm{link}}\) is what the layer multiplies with (e.g. in \(y = x\, W_{\mathrm{link}}\)).

Do **not** use \(w_{\mathrm{eff}}\) / “effective” in new text; prefer **link value**.

### 1.2 Ownership diagram

```text
Layer
 └── link (i, j)
      ├── swarm of S weights (bits)     ← discrete stored state
      ├── w_link = encoder(swarm)       ← matmul scalar
      └── g, m, v  (if used)            ← optimizer state per link
```

---

## 2. Locked design defaults

| Choice | Lock |
|--------|------|
| Matrix entry name | **Link** |
| Matmul scalar name | **Link value** \(w_{\mathrm{link}}\) |
| Where \(g, m, v\) attach | **Per link only** (not per weight-bit) |
| Default merge op \(\star\) | **XOR** (until explicitly changed) |
| Encoder input | **Only popcount / sum** of the swarm (no structured bit roles) |

### 2.1 Encoder (sum / popcount only)

With \(\pm 1\) weights \(a_1,\ldots,a_S\):

\[
s = \sum_{k=1}^{S} a_k \in [-S, S]
\quad\text{(steps of 2)},
\qquad
w_{\mathrm{link}} = \mathrm{enc}(s)
\]

Equivalent information: **popcount** of \(+1\) (or of set bits if \(0/1\) packing).

**Implications (frozen):**

- The encoder sees an **unordered** multiset of bits; only the sum matters.  
- Distinct swarms with the same sum yield the **same** link value.  
- At most \(S+1\) distinct sums (hence at most that many distinct link values for a
  monotone \(\mathrm{enc}\)).  
- **No** place-value / mantissa–exponent **roles inside** the swarm under this lock.  
- Any float-like curve must be a function of the **scalar** \(s\) only (e.g.
  \(w_{\mathrm{link}} = s/S\), or another 1D \(f(s)\)).

### 2.2 Optimizer and writeback (intended loop)

```text
swarm  --encoder-->  w_link  --forward/backward-->  g
 g, m, v  (per link)  --Adam/SGD-->  Δ
 Δ (+ noise)  --decoder-->  update swarm
 swarm  ←  swarm  XOR  update swarm
```

| Piece | Rule |
|-------|------|
| Continuous optimizer | Adam (\(g,m,v\)) or SGD (\(g\) / \(g,m\)) **on the link** |
| Decoder | Produces an update swarm of **\(S\)** bits from \(\Delta\) and noise |
| Merge | **XOR** (working default) |
| Intuition | Large \(\|\Delta\|\) → more bit disagreement applied; small \(\|\Delta\|\) → rare flips (noise) |

Exact decoder schedule and noise model are **implementation TBD**; names above are fixed.

---

## 3. Symbol sheet

| Symbol | Meaning |
|--------|---------|
| \(a_k \in \{\pm 1\}\) | \(k\)-th **weight** in the swarm |
| \(S\) | **Swarm size** |
| \(s = \sum_k a_k\) | Sum (encoder input) |
| \(w_{\mathrm{link}} = \mathrm{enc}(s)\) | **Link value** |
| \(g, m, v\) | Grad / moments **on the link** |
| \(\Delta\) | Continuous update on the link |
| \(u_k \in \{\pm 1\}\) | \(k\)-th bit of the **update swarm** |
| \(a \leftarrow a\ \mathrm{XOR}\ u\) | **Merge** (default \(\star\)) |

---

## 4. One-sentence stack

> Each **link** holds a **swarm** of \(S\) **weights**; the **encoder** maps
> **sum/popcount** of that swarm to a **link value** \(w_{\mathrm{link}}\) for the
> matmul; Adam/SGD keep \(g,m,v\) **on the link** and produce \(\Delta\); the
> **decoder** maps \(\Delta\) (+ noise) to an **update swarm**; **XOR** merges it
> into the weight swarm.

---

## 5. Map to legacy repo language

Use this table when reading older docs (`PLAN.md`, `SWARM_OPTIMIZER.md`, `v0_1`).

| Legacy term | This document |
|-------------|----------------|
| Agent | **Weight** (bit) |
| Population / per-connection swarm of agents | **Swarm** (on one **link**) |
| Logical weight / matrix entry / “connection” as weight | **Link** |
| Effective weight \(w\) / \(w_{\mathrm{eff}}\) from majority | **Link value** \(w_{\mathrm{link}}\) (different encoder in v0.1) |
| Swarm size \(S\) | **Swarm size** \(S\) (same symbol; meaning = bits per link) |
| Pressure EMA + stochastic flips | *Old update path* — not the XOR + decoder path here |
| Binary / register / place-value Swarm | **Out of scope** for this vocabulary (fixed-point-in-bits thesis) |

### 5.1 What `experiments/v0_1` does today (not this spec)

| Stage | This spec | Current `v0_1` |
|-------|-----------|----------------|
| Link value | \(\mathrm{enc}(s)\) multi-level from sum | \(\mathrm{sign}(s)\) → binary \(\pm 1\) |
| Optimizer | Adam/SGD per link | Flip rule + pressure EMA (no Adam on links) |
| Writeback | Decoder → update swarm → **XOR** | Flip agents disagreeing with \(-\mathrm{sign}(p)\) |

Implementations of **this** terminology should live under a new experiment id when built;
do not silently redefine v0.1 PROTOCOLs.

---

## 6. Words to avoid (in this design’s docs)

| Avoid | Prefer |
|-------|--------|
| Agent | **Weight** |
| Connection (for matrix entry) | **Link** |
| Effective weight / \(w_{\mathrm{eff}}\) | **Link value** / \(w_{\mathrm{link}}\) |
| Batch (for bits) | **Swarm** |
| Master weight \(W\) as stored state | (none — storage is the swarm) |

---

## 7. Changelog

| Date | Change |
|------|--------|
| 2026-08-05 | Initial freeze: weight / swarm / link / link value; sum-only encoder; per-link \(m,v\); XOR merge |
