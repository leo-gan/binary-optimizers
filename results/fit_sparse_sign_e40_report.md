# Fit-Scale Training Report (checkpointed)

- **Model:** bit_mlp  
- **Epochs / trials:** 40 / 1  
- **Device:** cpu  
- **Checkpoint root:** `checkpoints`  
- **Tag / schema:** `fit_v3` / v3  
- **Note:** Fit-scale MNIST with checkpoint cache. Retrain only if model/optimizer fingerprint changes or --force-retrain.

Checkpoints are reused unless model, optimizer, hyperparams, epochs, or seed change.

## Ranking

| Rank | Opt | Best | Final | Packed | Gap | Time | Cache |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `sparse_sign` | 0.9712 | 0.9674 | 0.9674 | +0.0131 | 233.0 | no |

**Best overall:** `sparse_sign`  
**Recommended default (Bit-MLP):** `ema_flip` — see `docs/optimizers.md`.

## Diagnostics

| Opt | Best | Final | Packed | Issues | Cache |
| :--- | ---: | ---: | ---: | :--- | :---: |
| `sparse_sign` | 0.9712 | 0.9674 | 0.9674 | — | no |

## Proposals

### 1. [HIGH] `docs` — Default Bit-MLP trainer is ema_flip

See docs/optimizers.md for reasons and trade-offs. Use --default-only or --optimizers ema_flip for focused runs; checkpoints under checkpoints/ (local only).

### 2. [HIGH] `cache` — Checkpoint cache is source of truth

Weights live under checkpoints/<slug>/. Re-run fit only with --force-retrain or after changing model/optimizer kwargs/schema. Use experiments/run_benchmark_checkpoints.py to re-evaluate only.

## Reproduce

```bash
# Train only missing fingerprints (or after optimizer/model changes)
uv run python experiments/run_fit_training.py
# Force retrain everything
uv run python experiments/run_fit_training.py --force-retrain
# Re-benchmark saved nets only (no training)
uv run python experiments/run_benchmark_checkpoints.py
```
