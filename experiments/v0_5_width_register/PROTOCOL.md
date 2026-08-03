# Experiment v0.5 — Width atlas: carry-safe binary **register**

**Path:** `experiments/v0_5_width_register/` · **Results:** `results/v0_5_width_register/`  
**Scaffold:** v0.3 carry-safe integer place-value (`experiments/v0_3/`)  
**Plan:** Question A — how much discrete state **per weight** (register width \(n\))?

## Claim

Vary only **register width** \(n_{\mathrm{bits}}\) on a fixed MNIST MLP and default train
settings. Answer whether accuracy saturates, peaks, or keeps rising as \(n\) goes
from small (8) toward large (up to 1024 if memory allows). **Do not** retune
hyperparameters per width (representation science, not tuning).

## Coding (fixed)

Same as v0.3: bits \(b_i\in\{0,1\}\), \(v=\sum b_i 2^i\), \(w=2v/(2^n-1)-1\),
carry-safe adaptive ±Δ on \(v\).

## Protocol

| Knob | Value |
|------|--------|
| Topology | Flatten → [LN] → Linear(784→H) → ReLU → [LN] → Linear(H→10) |
| `hidden` | 128 |
| `ln_mode` | default `none` (optional override) |
| Width grid | default `n_bits ∈ {8,16,32,48,62}` (int64-safe; CLI can request more but >62 is skipped) |
| Large \(n\) note | True integer \(v=0..2^n-1\) needs \(n\le 62\) for int64 encode/decode |
| Budget | max_epochs=80 **and** max_wall_sec=1200 (fair wall clock across widths) |
| Patience | patience_frac=0.125 of both budgets (~10 ep / 150s without gain) |
| min_delta | 0 (any strict test gain) |
| seed | 42 |
| Optimizer | v0.3 defaults; **max_step / step_scale scaled with \(2^n\)** so \(\Delta v / v_{\max}\) matches the n=16 reference (fair width comparison, not per-width accuracy search) |

## Deliverable

`results/v0_5_width_register/summary_*.json` with one row per width:
`n_bits`, `best_test_acc`, `epochs_ran`, `wall_sec`, `approx_state_bytes`.

## Naming

Human **v0.5 width-register** · path **`v0_5_width_register`**.
