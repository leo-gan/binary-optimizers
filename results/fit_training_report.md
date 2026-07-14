# Fit-Scale Training Report (checkpointed)

- **Model:** bit_mlp  
- **Epochs / trials:** 15 / 1  
- **Device:** cpu  
- **Checkpoint root:** `checkpoints`  
- **Tag / schema:** `fit_v2` / v2  
- **Note:** Fit-scale MNIST with checkpoint cache. Retrain only if model/optimizer fingerprint changes or --force-retrain.

Checkpoints are reused unless model, optimizer, hyperparams, epochs, or seed change.

## Ranking

| Rank | Opt | New | Best | Final | Packed | Gap | Time | Cache |
| :---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `adam` |  | 0.9609 | 0.9478 | 0.9478 | +0.0079 | 83.2 | yes |
| 2 | `ema_flip` | ★ | 0.9499 | 0.9487 | 0.9487 | -0.0060 | 89.0 | yes |
| 3 | `signum` |  | 0.9451 | 0.9381 | 0.9381 | +0.0038 | 78.0 | yes |
| 4 | `cosine_voting` | ★ | 0.9335 | 0.8803 | 0.8803 | +0.0519 | 84.1 | yes |
| 5 | `threshold_if` |  | 0.9082 | 0.8855 | 0.8855 | +0.0028 | 84.6 | yes |
| 6 | `sparse_sign` | ★ | 0.8910 | 0.8502 | 0.8502 | -0.0150 | 94.7 | yes |
| 7 | `ste` |  | 0.8847 | 0.8813 | 0.8813 | -0.0244 | 82.9 | yes |
| 8 | `voting` |  | 0.8240 | 0.6490 | 0.6490 | +0.1470 | 76.2 | yes |
| 9 | `hybrid_accumulator` | ★ | 0.7914 | 0.7580 | 0.7556 | +0.0022 | 87.8 | yes |

**Best baseline:** `adam`  
**Best new:** `ema_flip`

## Win matrix (best test)

| New | `adam` | `ste` | `voting` | `signum` | `threshold_if` | Wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ema_flip` | ❌ | ✅ | ✅ | ✅ | ✅ | **4/5** |
| `cosine_voting` | ❌ | ✅ | ✅ | ❌ | ✅ | **3/5** |
| `sparse_sign` | ❌ | ✅ | ✅ | ❌ | ❌ | **2/5** |
| `hybrid_accumulator` | ❌ | ❌ | ❌ | ❌ | ❌ | **0/5** |

## Diagnostics

| Opt | Best | Final | Packed | Issues | Cache |
| :--- | ---: | ---: | ---: | :--- | :---: |
| `adam` | 0.9609 | 0.9478 | 0.9478 | — | yes |
| `ema_flip` | 0.9499 | 0.9487 | 0.9487 | — | yes |
| `signum` | 0.9451 | 0.9381 | 0.9381 | — | yes |
| `cosine_voting` | 0.9335 | 0.8803 | 0.8803 | — | yes |
| `threshold_if` | 0.9082 | 0.8855 | 0.8855 | — | yes |
| `sparse_sign` | 0.8910 | 0.8502 | 0.8502 | — | yes |
| `ste` | 0.8847 | 0.8813 | 0.8813 | — | yes |
| `voting` | 0.8240 | 0.6490 | 0.6490 | overfit_gap, late_decay | yes |
| `hybrid_accumulator` | 0.7914 | 0.7580 | 0.7556 | — | yes |

## Proposals

### 1. [HIGH] `cache` — Checkpoint cache is source of truth

Weights live under checkpoints/<slug>/. Re-run fit only with --force-retrain or after changing model/optimizer kwargs/schema. Use experiments/run_benchmark_checkpoints.py to re-evaluate only.

### 2. [HIGH] `all_new` — Close gap to best baseline

Best new `ema_flip` (0.9499) vs baseline `adam` (0.9609). Continue Hybrid v2 ablations.

## Reproduce

```bash
# Train only missing fingerprints (or after optimizer/model changes)
uv run python experiments/run_fit_training.py
# Force retrain everything
uv run python experiments/run_fit_training.py --force-retrain
# Re-benchmark saved nets only (no training)
uv run python experiments/run_benchmark_checkpoints.py
```
