# Documentation

This folder contains documentation extracted from:

- `Benchmarking Voting Optimizer vs STE on CIFAR-10.md`

The docs are organized to capture:

- what each optimizer / layer does,
- why it was introduced in the notebook’s iteration history,
- how it improves on earlier variants,
- pros and cons.

## Contents

- `optimizers.md`
- `networks.md`
- `END_TO_END_SUMMARY.md` — inference engine, memory profiler, training sweep, Pareto analysis
- `NEW_OPTIMIZERS.md` — four new optimizers: rationale, win matrix, sequential proofs
- `FIT_TRAINING_ANALYSIS.md` — fit-scale MNIST training (full data, 15 epochs) + improvement proposals

## Checkpoints (trained nets)

Local cache under `checkpoints/` (gitignored). Fingerprint = model + optimizer +
hyperparams + epochs + seed + schema. Reuse unless those change.

```bash
uv run python experiments/run_fit_training.py          # train missing only
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
