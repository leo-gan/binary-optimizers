# Fit-Scale Training Report (checkpointed)

- **Model:** bit_mlp  
- **Epochs / trials:** 15 / 1  
- **Device:** cpu  
- **Checkpoint root:** `checkpoints`  
- **Tag / schema:** `fit_v3` / v3  
- **Note:** Fit-scale MNIST with checkpoint cache. Retrain only if model/optimizer fingerprint changes or --force-retrain.

Checkpoints are reused unless model, optimizer, hyperparams, epochs, or seed change.

## Ranking

| Rank | Opt | New | Best | Final | Packed | Gap | Time | Cache |
| :---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `ema_flip` | ★ | 0.9626 | 0.9614 | 0.9614 | +0.0083 | 100.0 | yes |
| 2 | `adam` |  | 0.9609 | 0.9478 | 0.9478 | +0.0079 | 79.9 | yes |
| 3 | `signum` |  | 0.9603 | 0.9578 | 0.9578 | +0.0058 | 85.9 | yes |
| 4 | `cosine_voting` | ★ | 0.9577 | 0.9577 | 0.9577 | +0.0035 | 85.4 | yes |
| 5 | `ste` |  | 0.9450 | 0.9450 | 0.9450 | -0.0101 | 76.2 | yes |
| 6 | `sparse_sign` | ★ | 0.8968 | 0.8852 | 0.8852 | -0.0152 | 86.1 | yes |
| 7 | `threshold_if` |  | 0.8813 | 0.8750 | 0.8750 | -0.0224 | 100.6 | yes |
| 8 | `voting` |  | 0.8625 | 0.8258 | 0.8258 | +0.0330 | 76.7 | yes |
| 9 | `hybrid_v2` | ★ | 0.8186 | 0.8186 | 0.8186 | -0.0143 | 89.1 | yes |
| 10 | `hybrid_accumulator` | ★ | 0.7839 | 0.7666 | 0.7713 | +0.0046 | 86.4 | yes |

**Best baseline:** `adam`  
**Best new:** `ema_flip`

## Win matrix (best test)

| New | `adam` | `ste` | `voting` | `signum` | `threshold_if` | Wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ema_flip` | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| `cosine_voting` | ❌ | ✅ | ✅ | ❌ | ✅ | **3/5** |
| `sparse_sign` | ❌ | ❌ | ✅ | ❌ | ✅ | **2/5** |
| `hybrid_accumulator` | ❌ | ❌ | ❌ | ❌ | ❌ | **0/5** |
| `hybrid_v2` | ❌ | ❌ | ❌ | ❌ | ❌ | **0/5** |

## Diagnostics

| Opt | Best | Final | Packed | Issues | Cache |
| :--- | ---: | ---: | ---: | :--- | :---: |
| `ema_flip` | 0.9626 | 0.9614 | 0.9614 | — | yes |
| `adam` | 0.9609 | 0.9478 | 0.9478 | — | yes |
| `signum` | 0.9603 | 0.9578 | 0.9578 | — | yes |
| `cosine_voting` | 0.9577 | 0.9577 | 0.9577 | — | yes |
| `ste` | 0.9450 | 0.9450 | 0.9450 | — | yes |
| `sparse_sign` | 0.8968 | 0.8852 | 0.8852 | — | yes |
| `threshold_if` | 0.8813 | 0.8750 | 0.8750 | — | yes |
| `voting` | 0.8625 | 0.8258 | 0.8258 | — | yes |
| `hybrid_v2` | 0.8186 | 0.8186 | 0.8186 | — | yes |
| `hybrid_accumulator` | 0.7839 | 0.7666 | 0.7713 | — | yes |

## Proposals

### 1. [HIGH] `cache` — Checkpoint cache is source of truth

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
