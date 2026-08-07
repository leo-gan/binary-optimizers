# Scripts

Helpers for **binary-optimizers**: fully binary/ternary latent-free training
(discrete NN + discrete optimizer). See the root [README](../README.md) and
[docs/PLAN.md](../docs/PLAN.md).

## `download_datasets.py`

Downloads **MNIST** and/or **CIFAR-10** into a local directory (default `./data`).

These files are **gitignored** and should never be committed. They are only
needed for local training / benchmarks — **not** for unit tests or CI.

```bash
# Install bench deps (torchvision), then download
uv sync --extra bench
python scripts/download_datasets.py

# MNIST only
python scripts/download_datasets.py --mnist --no-cifar
```

Experiment loaders already use `download=True`, so the first train run can
also fetch data; this script is for explicit offline setup and smaller surprises.

## `report.sh`

Prints the experiment comparison report from `results/experiments.duckdb`
(LayerNorm notes, version lineage, accuracy tables, simple analysis).

```bash
# From repo root
./scripts/report.sh

# Compact (no per-run detail table)
./scripts/report.sh --no-detail

# One experiment only
./scripts/report.sh --experiment v0_4

# Save markdown
./scripts/report.sh > results/report.md
```

Equivalent to: `uv run python -m binary_optimizers.store report …`.

Refresh the DB from JSON exports (if any) with:

```bash
uv run python -m binary_optimizers.store import
```

## `run_ste_vs_swarm.sh`

Shared-protocol MNIST comparison: STE (`ste_sgd`) vs Swarm v0.3 / v0.4.
Each run is logged to DuckDB (`experiment=ste_vs_swarm`). See
`experiments/ste_vs_swarm/PROTOCOL.md`.

```bash
# Full grid (long on CPU)
./scripts/run_ste_vs_swarm.sh

# Subset
./scripts/run_ste_vs_swarm.sh --methods ste_sgd,swarm_v0_3 --ln-mode affine

# Full lineage + STE vs Swarm report
./scripts/report.sh

# STE vs Swarm only
./scripts/report_ste_vs_swarm.sh
```

## `report_ste_vs_swarm.sh`

Prints only the **STE vs Swarm** analysis from DuckDB (method × LN table,
Swarm−STE deltas, per-run detail). Equivalent to:

`uv run python -m binary_optimizers.store report --experiment ste_vs_swarm`

```bash
./scripts/report_ste_vs_swarm.sh
./scripts/report_ste_vs_swarm.sh > results/ste_vs_swarm_report.md
```

If the section is empty, train first with `./scripts/run_ste_vs_swarm.sh`.
## `run_unary_ladder.sh` / `watch_unary_ladder.sh`

Sequential **Unary link Swarm** atlas (WP-U2 → U5): decoder ablations, width,
encoder family, CIFAR probe. Writes train stdout under gitignored `logs/`.

```bash
# Full ladder (long on CPU; override device with DEVICE=cuda)
bash scripts/run_unary_ladder.sh

# Optional: Grok monitor watcher (prints DONE or FAILED once)
bash scripts/watch_unary_ladder.sh
```

Results: `results/v0_9_*` … `v0_12_*` (gitignored). Notes:
`experiments/UNARY_LADDER_RESULTS.md`.
