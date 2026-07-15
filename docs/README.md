# Documentation

## Contents

- **`optimizers.md`** — unified optimizer guide: **default `ema_flip` for Bit-MLP**, design reasons, trade-offs, fit-scale comparison (all families in one place)
- `networks.md` — layers / architectures
- `END_TO_END_SUMMARY.md` — inference engine, memory profiler, training sweep, Pareto analysis
- `FIT_TRAINING_ANALYSIS.md` — fit-scale MNIST numbers and ranking
- `SPARSE_SIGN_MAX_FIT.md` — sparse_sign 40-epoch max-fit (97%+ test)
- `IMPROVEMENT_PROPOSALS.md` — research roadmap from fit_v3 (tiers, root causes, next steps)

Historical source notes (optional):

- `Benchmarking Voting Optimizer vs STE on CIFAR-10.md`

## Checkpoints (trained nets)

Local cache under `checkpoints/` (gitignored). Fingerprint = model + optimizer +
hyperparams + epochs + seed + schema. Reuse unless those change.

```bash
uv run python experiments/run_fit_training.py --include-swarm   # Bit-MLP + swarm combos
uv run python experiments/run_fit_training.py --swarm-only
uv run python experiments/run_fit_training.py --force-retrain
uv run python experiments/run_benchmark_checkpoints.py # eval only, no train
```

## Generated analysis artifacts

See `results/` (produced by `experiments/run_full_pipeline.py`):

- `inference_benchmark.md` / `.json`
- `training_sweep.json`
- `pareto_analysis.md` / `.json`
- `new_optimizers_report.md` / `.json`
- `SUMMARY.md`
