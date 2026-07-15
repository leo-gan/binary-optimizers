# Fit-Scale Training Report (checkpointed)

- **Model:** None  
- **Epochs / trials:** 15 / 1  
- **Device:** cpu  
- **Checkpoint root:** `checkpoints`  
- **Tag / schema:** `fit_v3` / v3  
- **Note:** Fit-scale MNIST with checkpoint cache. Includes Bit-MLP and swarm model×optimizer combinations. Retrain only if fingerprint changes or --force-retrain.

Checkpoints are reused unless model, optimizer, hyperparams, epochs, or seed change.

## Ranking

| Rank | Model | Opt | Best | Final | Packed | Gap | Time | Cache |
| :---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `swarm_mlp` | `adam` | 0.9703 | 0.9695 | 0.9695 | +0.0065 | 93.6 | no |
| 2 | `swarm_mlp` | `swarm_log_dynamic` | 0.9674 | 0.9658 | 0.9658 | +0.0048 | 176.9 | no |
| 3 | `bit_mlp` | `ema_flip` **(default)** | 0.9626 | 0.9614 | 0.9614 | +0.0083 | 100.0 | yes |
| 4 | `bit_mlp` | `adam` | 0.9609 | 0.9478 | 0.9478 | +0.0079 | 79.9 | yes |
| 5 | `bit_mlp` | `signum` | 0.9603 | 0.9578 | 0.9578 | +0.0058 | 85.9 | yes |
| 6 | `swarm_mlp` | `swarm_log` | 0.9593 | 0.9529 | 0.9529 | -0.0024 | 174.0 | no |
| 7 | `bit_mlp` | `sparse_sign` | 0.9591 | 0.9555 | 0.9555 | +0.0066 | 89.2 | yes |
| 8 | `bit_mlp` | `cosine_voting` | 0.9577 | 0.9577 | 0.9577 | +0.0035 | 85.4 | yes |
| 9 | `bit_mlp` | `ste` | 0.9450 | 0.9450 | 0.9450 | -0.0101 | 76.2 | yes |
| 10 | `swarm_mlp` | `swarm` | 0.9000 | 0.8944 | 0.8944 | -0.0045 | 134.4 | no |
| 11 | `bit_mlp` | `threshold_if` | 0.8813 | 0.8750 | 0.8750 | -0.0224 | 100.6 | yes |
| 12 | `bit_mlp` | `voting` | 0.8625 | 0.8258 | 0.8258 | +0.0330 | 76.7 | yes |
| 13 | `bit_mlp` | `hybrid_accumulator` | 0.7839 | 0.7666 | 0.7713 | +0.0046 | 86.4 | yes |

**Best overall:** `swarm_mlp+adam`  
**Recommended default (Bit-MLP):** `ema_flip` — see `docs/optimizers.md`.

Swarm rows use `swarm_mlp` (population bits); packed = majority-vote collapse.

## Diagnostics

| Model | Opt | Best | Final | Packed | Issues | Cache |
| :--- | :--- | ---: | ---: | ---: | :--- | :---: |
| `swarm_mlp` | `adam` | 0.9703 | 0.9695 | 0.9695 | — | no |
| `swarm_mlp` | `swarm_log_dynamic` | 0.9674 | 0.9658 | 0.9658 | — | no |
| `bit_mlp` | `ema_flip` | 0.9626 | 0.9614 | 0.9614 | — | yes |
| `bit_mlp` | `adam` | 0.9609 | 0.9478 | 0.9478 | — | yes |
| `bit_mlp` | `signum` | 0.9603 | 0.9578 | 0.9578 | — | yes |
| `swarm_mlp` | `swarm_log` | 0.9593 | 0.9529 | 0.9529 | — | no |
| `bit_mlp` | `sparse_sign` | 0.9591 | 0.9555 | 0.9555 | — | yes |
| `bit_mlp` | `cosine_voting` | 0.9577 | 0.9577 | 0.9577 | — | yes |
| `bit_mlp` | `ste` | 0.9450 | 0.9450 | 0.9450 | — | yes |
| `swarm_mlp` | `swarm` | 0.9000 | 0.8944 | 0.8944 | — | no |
| `bit_mlp` | `threshold_if` | 0.8813 | 0.8750 | 0.8750 | — | yes |
| `bit_mlp` | `voting` | 0.8625 | 0.8258 | 0.8258 | — | yes |
| `bit_mlp` | `hybrid_accumulator` | 0.7839 | 0.7666 | 0.7713 | — | yes |

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
