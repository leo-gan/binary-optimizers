# Scripts

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
