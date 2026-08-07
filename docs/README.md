# Documentation

## Grand goal (read this first)

This repository aims at **fully binary or fully ternary training**:

- a **neural net** whose stored weights are binary or ternary (latent-free; no FP master \(W\)), and  
- an **optimizer** that updates those weights with discrete rules (flips, votes, integer / trit steps).

That is the product ambition. Representation experiments (width, encoding, CIFAR
probes) exist to discover **laws** that make such a stack work—not to polish
floating-point baselines.

→ Project overview: **[../README.md](../README.md)**  
→ Living roadmap: **[PLAN.md](PLAN.md)**

---

## Core docs (current research)

| Doc | Purpose |
|-----|---------|
| **[SWARM_OPTIMIZER.md](SWARM_OPTIMIZER.md)** | **Swarm optimizer:** terminology, codings, updates, trade-offs, reasonings |
| **[UNARY_SWARM_TERMINOLOGY.md](UNARY_SWARM_TERMINOLOGY.md)** | **Frozen** Unary Swarm vocabulary (weight / swarm / link / link value; XOR path) |
| **[UNARY_SWARM_EXPERIMENT_PLAN.md](UNARY_SWARM_EXPERIMENT_PLAN.md)** | Unary Swarm experiment ladder (WP-U0–U5, `v0_8`…`v0_12`) |
| [PLAN.md](PLAN.md) | Roadmap: existence proofs, WP1 width, WP2 encoding, WP3 scale |
| [OPTIMA_STATUS.md](OPTIMA_STATUS.md) | Sketch optima (register \(n\), unary \(S\), encodings, CIFAR) |
| **[MEMORY_1B.md](MEMORY_1B.md)** | Static training memory at **1B** params: FP / STE / binary Swarm / ternary Swarm |
| [TRAIN_BUDGET.md](TRAIN_BUDGET.md) | Pure wall-clock train protocol (`pure_wall_budget_v1`) |
| [EXPERIMENT_VERSIONS.md](EXPERIMENT_VERSIONS.md) | Experiment IDs when the train protocol changes |

---

## Historical / reference extracts

Material distilled from earlier notebook work:

- [optimizers.md](optimizers.md) — optimizer variants and motivation  
- [networks.md](networks.md) — network sketches (incl. CIFAR SmallConvNet notes)  
- [Benchmarking Voting Optimizer vs STE on CIFAR-10.md](Benchmarking%20Voting%20Optimizer%20vs%20STE%20on%20CIFAR-10.md)  

These inform STE / voting baselines; the **latent-free Swarm ladder** lives under
`experiments/v0_*` and is governed by PLAN.md.

---

## Experiments (pointer)

| Path | Role |
|------|------|
| `experiments/v0_1` … `v0_4` | Existence: unary → place-value → carry-safe → ternary |
| `experiments/v0_5_width_*` | WP1 width atlas |
| `experiments/v0_6_encoding` | WP2 encoding atlas (MNIST) |
| `experiments/v0_7_cifar_encoding` | WP3 sparse fixed vs exp/mant on CIFAR-10 |
| `experiments/v0_8_unary_link` … `v0_12_unary_cifar` | **Unary link Swarm** ladder (sum→link value, XOR writeback) |
| `experiments/UNARY_LADDER_RESULTS.md` | Consolidated CPU pure-wall ladder results (2026-08-06) |
| `experiments/ste_vs_swarm` | Optional STE vs Swarm harness (not the main goal) |

Each folder has `PROTOCOL.md` / `README.md`. Notes: `experiments/v0_5_NOTES.md`,
`v0_6_NOTES.md`, `v0_7_NOTES.md`.

---

## Temp research notes

`docs/temp/` — working notes and deeper background (not the source of truth for
status; prefer PLAN.md and the README).
