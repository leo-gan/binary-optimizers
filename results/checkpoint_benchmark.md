# Checkpoint Benchmark (no training)

Model `bit_mlp`, epochs=15, seed=42

| Optimizer | STE test | Packed test | Cached best | Status |
| :--- | ---: | ---: | ---: | :--- |
| `adam` | 0.9478 | 0.9478 | 0.9609 | ok |
| `ste` | 0.9450 | 0.9450 | 0.945 | ok |
| `voting` | 0.8258 | 0.8258 | 0.8625 | ok |
| `signum` | 0.9578 | 0.9578 | 0.9603 | ok |
| `threshold_if` | 0.8750 | 0.8750 | 0.8813 | ok |
| `ema_flip` | 0.9614 | 0.9614 | 0.9626 | ok |
| `cosine_voting` | 0.9577 | 0.9577 | 0.9577 | ok |
| `sparse_sign` | 0.8852 | 0.8852 | 0.8968 | ok |
| `hybrid_accumulator` | 0.7666 | 0.7713 | 0.7839 | ok |
| `hybrid_v2` | 0.8186 | 0.8186 | 0.8186 | ok |
