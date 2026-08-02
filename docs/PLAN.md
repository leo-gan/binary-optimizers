# Swarm / latent-free training — roadmap

**Branch:** `plan/swarm-roadmap`  
**Last updated:** 2026-07-31  
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

### 2.4 Incomplete / weak spots

| Item | Status |
|------|--------|
| Full **STE vs Swarm** matrix (all methods × LN) | **Incomplete** — store only has `ste_sgd_ln_none` (~0.21 test, likely mis-tuned / aborted) |
| Multi-seed error bars | Not run (all seed=42) |
| `n_bits` / `n_trits` Pareto | Not run |
| CIFAR / Transformer | Not started |
| Pack int8 → bitplanes for memory claims | Prototype only in old inference code |
| CI without datasets | Policy agreed; ensure CI only runs unit tests |

---

## 3. Architecture of the current “winner”

**v0.3 (default recommendation for next baselines)**

```
Flatten → [LN?] → CarrySafeLinear(784→128) → ReLU → [LN?] → CarrySafeLinear(128→10)
```

- Bits \(b_i\in\{0,1\}\), \(v=\sum b_i 2^i\), \(w=2v/(2^n-1)-1\).
- Optimizer: EMA of \(\partial L/\partial w\); with prob ∝ \|EMA\|, apply adaptive integer Δ and **re-encode** (carries exact).
- Optional LN affine params via small SGD group.

**v0.4** same topology with balanced ternary digits and places \(3^i\).

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

3. **Pareto: bits/param vs accuracy**  
   - v0.3: `n_bits ∈ {4,8,12,16,24}`  
   - v0.4: `n_trits ∈ {4,6,8,10,12}`  
   - X-axis: agent bits (or log₂(levels)); Y-axis: test acc.  
4. **Memory accounting**  
   - Report true storage if packed (1 bit/agent, ~1.6 bit/trit) vs float±1 prototype.  
   - Optional: pack/unpack in checkpoint without changing training math.

**Exit:** curve showing where carry-safe Swarm beats 8-bit Adam / STE on bits/param story (or not).

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
