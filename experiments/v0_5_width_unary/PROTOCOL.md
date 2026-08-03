# Experiment v0.5 — Width atlas: **unary** Swarm population size

**Path:** `experiments/v0_5_width_unary/` · **Results:** `results/v0_5_width_unary/`  
**Scaffold:** v0.1 unary majority Swarm (`experiments/v0_1/`)  
**Plan:** Question A — population size \(S\) per weight (not mixed with register \(n\)).

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
| epochs / patience | 80 / **5** (early stop; atlas ranking, not polish) |
| seed | 42 |
| Optimizer | v0.1 defaults (`recruit_rate=1e4`, …) |

## Deliverable

`results/v0_5_width_unary/summary_*.json` with one row per \(S\).

## Naming

Human **v0.5 width-unary** · path **`v0_5_width_unary`**.
