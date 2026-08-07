# binary-optimizers

## Grand goal

**Make training fully binary or fully ternary** — not only at inference, and not only
by quantizing a floating-point master weight.

That means both of:

1. **Network (NN)** — weights (and eventually more of the stack) live in a
   **discrete** state: binary agents/bits or ternary trits, not an FP32 matrix \(W\).
2. **Optimizer** — updates are **discrete** (flips, majority votes, place-value /
   carry-safe integer steps), not SGD/Adam on a latent full-precision copy of \(W\).

This is **latent-free** training: the discrete register *is* the weight. That is
stricter than BitNet-style QAT / STE, which typically keeps a full-precision \(W\)
and only uses low-bit values in the forward (or for a simulated discrete step).

**North star:** a stack where the learnable parameters and the update rule are
binary or ternary end-to-end. Autograd may still use floating point for *pressure*
signals; the stored state we train is discrete.

---

## Why this project

| Approach | Weight state | Optimizer |
|----------|--------------|-----------|
| Standard deep learning | FP | FP (Adam, SGD, …) |
| Quantization / BitNet QAT / STE | FP master + discrete *view* | FP on master \(W\) |
| **This work (target)** | **Binary / ternary only** | **Binary / ternary updates** |

Research progress is organized as experiments that answer representation questions
(how wide the discrete state is, how it is encoded) while holding the latent-free
constraint. See **[docs/PLAN.md](docs/PLAN.md)**.

---

## Current status (roadmap)

| Phase | What | Status |
|-------|------|--------|
| v0.1–v0.4 | Existence proofs: unary → place-value → carry-safe → ternary Swarm on MNIST | **Done** |
| WP1 | Width laws: register \(n\), unary \(S\) | **Done** (sketch): \(n\approx 8\); \(S\) plateaus ~256–1024 |
| WP2 | Encoding: fixed-point vs exp/mant vs block scale | **Done** (sketch): fixed-point wins on MNIST |
| WP3 | Do optima move? (CIFAR / scale) | **Sparse probe done**: CIFAR flat MLP still ranks fixed > exp/mant |
| Later | Deeper nets, stronger vision models, less FP in the *signal* path | Open |

**Default scaffold today:** carry-safe binary register (v0.3 lineage) for register /
encoding work; unary Swarm (v0.1) for population-size studies; ternary (v0.4) when
trit structure is the object of study.

Optima snapshot: [docs/OPTIMA_STATUS.md](docs/OPTIMA_STATUS.md).  
Train protocol (pure wall budget): [docs/TRAIN_BUDGET.md](docs/TRAIN_BUDGET.md).  
Experiment IDs / re-run versioning: [docs/EXPERIMENT_VERSIONS.md](docs/EXPERIMENT_VERSIONS.md).

---

## Repository layout

```
binary_optimizers/   # library: optimizers, models, data, store, training budgets
experiments/         # versioned research runs (v0_1 … v0_7_*)
docs/                # roadmap, protocol notes, historical notebooks
scripts/             # dataset download, report helpers
results/             # local run outputs (gitignored)
```

Semantic experiment paths: `experiments/v0_N/` or `experiments/v0_N_<topic>/`.

---

## Quick start

```bash
# Install (CPU torch index via uv; see pyproject.toml)
uv sync --extra bench

# Data (local only; not committed)
python scripts/download_datasets.py --mnist --cifar

# Example: latent-free register reference (v0.3)
python experiments/v0_3/train.py --ln-mode none

# Width atlas (WP1)
python experiments/v0_5_width_register/train.py --ln-mode none --seed 42
python experiments/v0_5_width_unary/train.py --ln-mode none --seed 42

# Encoding atlas (WP2)
python experiments/v0_6_encoding/train.py --ln-mode none --seed 42

# CIFAR sparse encoding probe (WP3)
python experiments/v0_7_cifar_encoding/train.py --ln-mode none --seed 42
```

Tests:

```bash
uv run pytest experiments/v0_1 experiments/v0_6_encoding experiments/v0_7_cifar_encoding tests -q
```

---

## Documentation map

| Doc | Content |
|------|---------|
| **[docs/SWARM_OPTIMIZER.md](docs/SWARM_OPTIMIZER.md)** | **Swarm in depth:** terminology, codings, updates, trade-offs |
| **[docs/UNARY_SWARM_TERMINOLOGY.md](docs/UNARY_SWARM_TERMINOLOGY.md)** | Frozen Unary Swarm terms (weight / swarm / link / XOR path) |
| **[docs/UNARY_SWARM_EXPERIMENT_PLAN.md](docs/UNARY_SWARM_EXPERIMENT_PLAN.md)** | Unary Swarm experiment plan (WP-U0–U5) |
| **[docs/PLAN.md](docs/PLAN.md)** | Living research roadmap and work packages |
| **[docs/OPTIMA_STATUS.md](docs/OPTIMA_STATUS.md)** | Width / encoding / CIFAR sketch results |
| **[docs/MEMORY_1B.md](docs/MEMORY_1B.md)** | Static training memory @ 1B params (FP / STE / Swarm) |
| **[docs/TRAIN_BUDGET.md](docs/TRAIN_BUDGET.md)** | Pure wall-clock train protocol |
| **[docs/EXPERIMENT_VERSIONS.md](docs/EXPERIMENT_VERSIONS.md)** | Run IDs vs protocol revs (e.g. `v0_2` → `v0_2_1`) |
| [docs/README.md](docs/README.md) | Doc index (incl. older optimizer notes) |
| [docs/optimizers.md](docs/optimizers.md) / [docs/networks.md](docs/networks.md) | Historical notebook extract |

---

## What is *not* the goal (yet)

- Shaving leaderboard points with multi-seed STE vs Swarm contests  
- Full LLM pretrain  
- Pure integer *backward* / bit-only autograd (longer-term; discrete **state** is the near-term goal)  
- Committing dataset blobs or large result dumps  

---

## License / package

Python package name: `binary-optimizers` (`pyproject.toml`). Research code under
`experiments/` is the primary vehicle for the fully binary/ternary training goal.
