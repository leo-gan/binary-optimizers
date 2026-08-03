# Experiment v0.5 — Width atlas: **unary** Swarm population size

**Path:** `experiments/v0_5_width_unary/`  
**Run / DB id:** `v0_5_1_width_unary` (parent `v0_5_width_unary`; wall-budget protocol)  
**Results:** `results/v0_5_1_width_unary/`  
**Scaffold:** v0.1 unary majority Swarm (`experiments/v0_1/`)  
**Plan:** Question A — population size \(S\) per weight (not mixed with register \(n\)).  
See `docs/EXPERIMENT_VERSIONS.md`.

## Claim

Vary only **swarm size** \(S\) (number of equal ±1 agents per weight) on a fixed
MNIST MLP and default train settings. Map accuracy vs \(S\) for the **unary**
coding family. Do not retune hyperparameters per \(S\).

## Coding (fixed)

v0.1 unary: \(S\) agents in \(\{-1,+1\}\), equal weight; effective weight from
majority / STE through normalized sum.

## Protocol

| Knob | Value |
|------|--------|
| Topology | Same BitNet-style MLP as v0.1 |
| `hidden` | 128 |
| `ln_mode` | default `none` |
| Width grid | \(S \in \{8,16,32,64,128,256,512,1024\}\) (skip if OOM) |
| Budget | max_epochs=80 **and** max_wall_sec=1200 (fair wall clock across \(S\)) |
| Patience | patience_frac=0.125 of both budgets (~10 ep / 150s without gain) |
| min_delta | 0 (any strict test gain) |
| seed | 42 |
| Optimizer | v0.1 defaults (`recruit_rate=1e4`, …) |

## Deliverable

`results/v0_5_width_unary/summary_*.json` with one row per \(S\).

## Naming

Human **v0.5 width-unary** · path **`v0_5_width_unary`**.
