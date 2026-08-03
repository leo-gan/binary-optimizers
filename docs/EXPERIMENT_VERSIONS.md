# Experiment version IDs (protocol revisions)

Code lives under `experiments/v0_N…/`. The **experiment id** written to
`results/<id>/` and DuckDB (`runs.experiment`) is bumped when the **train
protocol** changes so re-runs are not mixed with legacy numbers.

| New id (re-run) | Parent (legacy) | Protocol |
|-----------------|-----------------|----------|
| `v0_1_1` | `v0_1` | `wall_epoch_budget_v1` |
| `v0_2_1` | `v0_2` | same |
| `v0_3_1` | `v0_3` | same |
| `v0_4_1` | `v0_4` | same |
| `v0_5_1_width_register` | `v0_5_width_register` | same |
| `v0_5_1_width_unary` | `v0_5_width_unary` | same |
| `v0_6_1_encoding` | `v0_6_encoding` | same |
| `ste_vs_swarm_1` | `ste_vs_swarm` | same |

**`wall_epoch_budget_v1`:** stop on `max_wall_sec` (default 1200) or `max_epochs`
(default 80); patience = `patience_frac` (default 0.125) of **both** budgets;
`min_delta` default 0.

**Why these numbers:** see **`docs/TRAIN_BUDGET.md`** (fairness across fast vs
slow epochs, 20 min wall policy, fractional patience).

Registry source of truth: `binary_optimizers/store/versions.py`.

## DuckDB

Each completed re-run should store:

- `runs.experiment` = new id (e.g. `v0_2_1`)
- `runs.notes` = parent + protocol + changelog
- `runs.config` JSON includes `experiment_id`, `experiment_parent`,
  `train_protocol`, `protocol_changelog`, `code_dir`, and train `budget`

```bash
python -m binary_optimizers.store list --experiment v0_2_1
python -m binary_optimizers.store best --experiment v0_2_1
```

Legacy parent ids remain importable from old `results/v0_*/` JSON.
