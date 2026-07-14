# Checkpoint Benchmark (no training)

Model `bit_mlp`, epochs=15, seed=42

| Optimizer | STE test | Packed test | Cached best | Status |
| :--- | ---: | ---: | ---: | :--- |
| `adam` | 0.9478 | 0.9478 | 0.9609 | ok |
| `ste` | 0.8813 | 0.8813 | 0.8847 | ok |
| `voting` | 0.6490 | 0.6490 | 0.824 | ok |
| `signum` | 0.9381 | 0.9381 | 0.9451 | ok |
| `threshold_if` | 0.8855 | 0.8855 | 0.9082 | ok |
| `ema_flip` | 0.9487 | 0.9487 | 0.9499 | ok |
| `cosine_voting` | 0.8803 | 0.8803 | 0.9335 | ok |
| `sparse_sign` | 0.8502 | 0.8502 | 0.891 | ok |
| `hybrid_accumulator` | 0.7580 | 0.7556 | 0.7914 | ok |
