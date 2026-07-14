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

## Generated analysis artifacts

See `results/` (produced by `experiments/run_full_pipeline.py`):

- `inference_benchmark.md` / `.json`
- `training_sweep.json`
- `pareto_analysis.md` / `.json`
- `SUMMARY.md`
