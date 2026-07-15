# SparseSign max-fit run (40 epochs)

Extended training to fully fit Bit-MLP with `sparse_sign` after the 15-epoch suite under-fit (~87% train / ~90% test).

## Recipe changes

| Knob | Previous (e15) | Max-fit (e40) |
| :--- | :--- | :--- |
| Epochs | 15 | **40** |
| Density | anneal 1.0→0.45 from step 0 | **hold 1.0 for 75% of steps**, then → 0.55 |
| LR | fixed 0.05 | **cosine 0.06 → 0.005** |
| Confidence | 0.001 | **0** (update all dense positions) |
| BN | dual Adam | dual Adam (unchanged) |

Checkpoint: `checkpoints/fit_v3_bit_mlp_sparse_sign_e40_s42_*` (local only).

## Results (MNIST Bit-MLP h=128, seed 42)

| Metric | e15 (suite) | **e40 max-fit** |
| :--- | ---: | ---: |
| Best test | 0.8968 | **0.9712** |
| Final test | 0.8852 | **0.9674** |
| Train final | ~0.870 | **0.9805** |
| Packed (±1) | 0.8852 | **0.9674** |
| Wall time | ~86 s | ~230 s |

### Curve (test acc)

Dense phase (ep 1–30) climbs from ~0.85 → **0.96+**; late sparse+low-lr phase holds **~0.97**.

Peak at epoch **37** (0.9712); last epoch 0.9674 (−0.4 pp).

## Comparison to suite leaders (15-ep protocol)

| Optimizer | Epochs | Best test | Final test |
| :--- | ---: | ---: | ---: |
| **sparse_sign** | **40** | **0.9712** | **0.9674** |
| ema_flip | 15 | 0.9626 | 0.9614 |
| adam | 15 | 0.9609 | 0.9478 |
| signum | 15 | 0.9603 | 0.9578 |

**Caveat:** Not matched on epoch budget. SparseSign needed **longer dense training** to enter the same accuracy band; at equal 15 epochs it lagged badly.

## Trade-offs

| Pros | Cons |
| :--- | :--- |
| Fully fits when density stays high long enough | ~2.7× wall time vs 15-ep suite |
| Late sparsity does not destroy the fit | Matched-epoch comparisons need e40 for fair sparse |
| Packed = STE | More hyperparameters (hold frac, anneal end) |

## Reproduce

```bash
uv run python experiments/run_fit_training.py \
  --optimizers sparse_sign --epochs 40 --seed 42
# later (no retrain):
uv run python experiments/run_benchmark_checkpoints.py \
  --optimizers sparse_sign --epochs 40
```

Artifacts: `results/fit_sparse_sign_e40.json`, `results/fit_sparse_sign_e40_report.md`.
