# Experiment v0.6 — Encoding atlas (Question B)

**Path:** `experiments/v0_6_encoding/`  
**Run / DB id:** `v0_6_1_encoding` (parent `v0_6_encoding`; wall-budget protocol)  
**Results:** `results/v0_6_1_encoding/`  
**Plan:** WP2 — fixed total bit budget \(n\); vary **encoding structure** only.  
**Scaffold:** register / carry-safe lineage (not unary digits).  
See `docs/EXPERIMENT_VERSIONS.md`.

## Claim

At budgets locked from WP1, rank **encoding classes** (fixed-point vs exp+mantissa
vs block scale). Unary \(S=256\) is a **baseline**, not an encoding design space.

## Budgets (from WP1)

| \(n\) | Role |
|------:|------|
| **8** | Primary (register local optimum) |
| **16** | Primary (still healthy fixed-point) |
| **32** | Optional `--include-rescue` (cliff under pure fixed-point) |

## Encodings

| ID | Decode (sketch) |
|----|-----------------|
| `fixed` | \(w = 2v/(2^n-1)-1\) (v0.3) |
| `exp_mant:ne` | Per-weight: \(n_e\) exp + \(n_m=n-n_e\) mant; \(w = m_{\mathrm{norm}}\cdot 2^{e-b}/s_{\max}\) |
| `block_scale:ne` | Full \(n\) mant bits + shared row exponent \(n_e\) bits |
| `unary:S` | v0.1 majority baseline (default \(S=256\)) |

Default structure cells: `fixed`, `exp_mant` with \(n_e\in\{2,3,4\}\) (when \(n_e<n\)),
`block_scale:3`, plus unary baseline once.

## Protocol

| Knob | Value |
|------|--------|
| Topology | Same MNIST MLP as v0.5 (`hidden=128`) |
| `ln_mode` | default `none` |
| Budget | **max_epochs=80** and **max_wall_sec=1200** (whichever hits first) |
| Patience | **patience_frac=0.125** → ~10 epochs **and** 150s wall without gain |
| min_delta | **0** (any strict test gain resets stall) |
| seed | 42 |
| Mantissa steps | Δv/vmax scaled to n=16 reference (same spirit as v0.5 register) |
| Exp steps | small (`exp_max_step=1`) |

**No per-cell hyperparam search.**

## Deliverable

`results/v0_6_encoding/summary_*.json` + curve CSV: one row per cell
`(n_bits or S, encoding, best_test_acc, …)` plus weight/exp stats when relevant.

## Naming

Human **v0.6 encoding** · path **`v0_6_encoding`**.
