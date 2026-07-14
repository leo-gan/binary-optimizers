# Pareto Analysis Report

Combines scaffolding training-sweep results with inference benchmarks.

**Note:** Scaffolding run — short epochs/batches to demonstrate pipeline. bit_mlp_small configs use extended scaffold (6 epochs, 8 train batches) so new-optimizer warm-up is visible; other configs remain 2-epoch micro-scaffold.

## Full comparison

| Config | Dataset | Test Acc (mean ± std) | Infer mem (bitpacked) | Infer latency (ms) | Train mem | Train time (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| mnist_small_ste | mnist | 0.5371 ± 0.1740 | 6.4 KB | 0.4118 | 407.6 KB | 0.41 |
| mnist_small_threshold_if | mnist | 0.4561 ± 0.0014 | 6.4 KB | 0.4118 | 407.0 KB | 0.43 |
| mnist_small_voting | mnist | 0.3164 ± 0.0663 | 6.4 KB | 0.4118 | 407.0 KB | 0.45 |
| mnist_swarm | mnist | 0.2695 ± 0.0387 | 6.4 KB | 0.8824 | 1.63 MB | 0.07 |
| mnist_small_signum | mnist | 0.2119 ± 0.0014 | 6.4 KB | 0.4118 | 407.0 KB | 0.39 |
| mnist_large_ste | mnist | 0.1992 ± 0.0552 | 14.8 KB | 0.4118 | 948.2 KB | 0.09 |
| mnist_large_threshold_if | mnist | 0.1992 ± 0.0442 | 14.8 KB | 0.4118 | 946.2 KB | 0.06 |
| mnist_small_ema_flip | mnist | 0.1953 ± 0.0193 | 6.4 KB | 0.4118 | 407.0 KB | 0.44 |
| mnist_large_voting | mnist | 0.1953 ± 0.0055 | 14.8 KB | 0.4118 | 946.2 KB | 0.10 |
| mnist_small_sparse_sign | mnist | 0.1924 ± 0.0014 | 6.4 KB | 0.4118 | 407.0 KB | 0.44 |
| mnist_large_sparse_sign | mnist | 0.1387 ± 0.0580 | 14.8 KB | 0.4118 | 946.2 KB | 0.07 |
| cifar_adam | cifar10 | 0.1348 ± 0.0083 | 262.5 KB | 0.0993 | 26.11 MB | 0.51 |
| cifar_signum | cifar10 | 0.1328 ± 0.0000 | 262.5 KB | 0.0993 | 17.40 MB | 0.51 |
| cifar_ste_sgd | cifar10 | 0.1172 ± 0.0000 | 131.4 KB | 0.0993 | 8.57 MB | 0.25 |
| mnist_small_cosine_voting | mnist | 0.1104 ± 0.0401 | 6.4 KB | 0.4118 | 407.0 KB | 0.44 |
| mnist_small_adam | mnist | 0.0820 ± 0.0000 | 6.4 KB | 0.4118 | 611.3 KB | 0.48 |
| mnist_small_hybrid_accumulator | mnist | 0.0820 ± 0.0000 | 6.4 KB | 0.4118 | 610.3 KB | 0.41 |
| mnist_large_ema_flip | mnist | 0.0742 ± 0.0000 | 14.8 KB | 0.4118 | 946.2 KB | 0.13 |
| mnist_large_cosine_voting | mnist | 0.0742 ± 0.0000 | 14.8 KB | 0.4118 | 946.2 KB | 0.06 |
| mnist_large_hybrid_accumulator | mnist | 0.0742 ± 0.0000 | 14.8 KB | 0.4118 | 1.42 MB | 0.06 |
| mnist_large_signum | mnist | 0.0723 ± 0.0028 | 14.8 KB | 0.4118 | 946.2 KB | 0.06 |
| mnist_large_adam | mnist | 0.0703 ± 0.0055 | 14.8 KB | 0.4118 | 1.42 MB | 0.06 |

## Accuracy vs inference memory

| Config | Acc | Bitpacked mem |
| :--- | ---: | ---: |
| mnist_small_adam | 0.0820 | 6.4 KB |
| mnist_small_ste ★ | 0.5371 | 6.4 KB |
| mnist_small_voting | 0.3164 | 6.4 KB |
| mnist_small_signum | 0.2119 | 6.4 KB |
| mnist_small_threshold_if | 0.4561 | 6.4 KB |
| mnist_small_ema_flip | 0.1953 | 6.4 KB |
| mnist_small_cosine_voting | 0.1104 | 6.4 KB |
| mnist_small_sparse_sign | 0.1924 | 6.4 KB |
| mnist_small_hybrid_accumulator | 0.0820 | 6.4 KB |
| mnist_swarm | 0.2695 | 6.4 KB |
| mnist_large_adam | 0.0703 | 14.8 KB |
| mnist_large_ste | 0.1992 | 14.8 KB |
| mnist_large_voting | 0.1953 | 14.8 KB |
| mnist_large_signum | 0.0723 | 14.8 KB |
| mnist_large_threshold_if | 0.1992 | 14.8 KB |
| mnist_large_ema_flip | 0.0742 | 14.8 KB |
| mnist_large_cosine_voting | 0.0742 | 14.8 KB |
| mnist_large_sparse_sign | 0.1387 | 14.8 KB |
| mnist_large_hybrid_accumulator | 0.0742 | 14.8 KB |
| cifar_ste_sgd | 0.1172 | 131.4 KB |
| cifar_adam | 0.1348 | 262.5 KB |
| cifar_signum | 0.1328 | 262.5 KB |

## Accuracy vs inference speed

| Config | Acc | Latency (ms) |
| :--- | ---: | ---: |
| cifar_adam ★ | 0.1348 | 0.0993 |
| cifar_signum | 0.1328 | 0.0993 |
| cifar_ste_sgd | 0.1172 | 0.0993 |
| mnist_small_adam | 0.0820 | 0.4118 |
| mnist_small_ste ★ | 0.5371 | 0.4118 |
| mnist_small_voting | 0.3164 | 0.4118 |
| mnist_small_signum | 0.2119 | 0.4118 |
| mnist_small_threshold_if | 0.4561 | 0.4118 |
| mnist_small_ema_flip | 0.1953 | 0.4118 |
| mnist_small_cosine_voting | 0.1104 | 0.4118 |
| mnist_small_sparse_sign | 0.1924 | 0.4118 |
| mnist_small_hybrid_accumulator | 0.0820 | 0.4118 |
| mnist_large_adam | 0.0703 | 0.4118 |
| mnist_large_ste | 0.1992 | 0.4118 |
| mnist_large_voting | 0.1953 | 0.4118 |
| mnist_large_signum | 0.0723 | 0.4118 |
| mnist_large_threshold_if | 0.1992 | 0.4118 |
| mnist_large_ema_flip | 0.0742 | 0.4118 |
| mnist_large_cosine_voting | 0.0742 | 0.4118 |
| mnist_large_sparse_sign | 0.1387 | 0.4118 |
| mnist_large_hybrid_accumulator | 0.0742 | 0.4118 |
| mnist_swarm | 0.2695 | 0.8824 |

## Training efficiency (acc vs train mem & time)

| Config | Acc | Train mem | Time (s) |
| :--- | ---: | ---: | ---: |
| mnist_large_adam ★ | 0.0703 | 1.42 MB | 0.06 |
| mnist_large_signum ★ | 0.0723 | 946.2 KB | 0.06 |
| mnist_large_threshold_if ★ | 0.1992 | 946.2 KB | 0.06 |
| mnist_large_cosine_voting | 0.0742 | 946.2 KB | 0.06 |
| mnist_large_hybrid_accumulator | 0.0742 | 1.42 MB | 0.06 |
| mnist_large_sparse_sign | 0.1387 | 946.2 KB | 0.07 |
| mnist_swarm ★ | 0.2695 | 1.63 MB | 0.07 |
| mnist_large_ste | 0.1992 | 948.2 KB | 0.09 |
| mnist_large_voting | 0.1953 | 946.2 KB | 0.10 |
| mnist_large_ema_flip | 0.0742 | 946.2 KB | 0.13 |
| cifar_ste_sgd | 0.1172 | 8.57 MB | 0.25 |
| mnist_small_signum ★ | 0.2119 | 407.0 KB | 0.39 |
| mnist_small_ste ★ | 0.5371 | 407.6 KB | 0.41 |
| mnist_small_hybrid_accumulator | 0.0820 | 610.3 KB | 0.41 |
| mnist_small_threshold_if ★ | 0.4561 | 407.0 KB | 0.43 |
| mnist_small_sparse_sign | 0.1924 | 407.0 KB | 0.44 |
| mnist_small_ema_flip | 0.1953 | 407.0 KB | 0.44 |
| mnist_small_cosine_voting | 0.1104 | 407.0 KB | 0.44 |
| mnist_small_voting | 0.3164 | 407.0 KB | 0.45 |
| mnist_small_adam | 0.0820 | 611.3 KB | 0.48 |
| cifar_signum | 0.1328 | 17.40 MB | 0.51 |
| cifar_adam | 0.1348 | 26.11 MB | 0.51 |

## Pareto-optimal configurations

- **Accuracy vs inference memory:** mnist_small_ste
- **Accuracy vs inference speed:** mnist_small_ste, cifar_adam
- **Training efficiency:** mnist_small_ste, mnist_small_signum, mnist_small_threshold_if, mnist_large_adam, mnist_large_signum, mnist_large_threshold_if, mnist_swarm
- **All objectives:** mnist_small_ste, mnist_small_signum, mnist_small_threshold_if, mnist_large_adam, mnist_large_signum, mnist_large_threshold_if, mnist_swarm, cifar_adam, cifar_signum, cifar_ste_sgd

★ marks Pareto-optimal points on the corresponding frontier.

## Interpretation

- Higher accuracy with lower bitpacked memory is preferred for edge deployment.
- Swarm configs store a population during training but can collapse to a majority
  vote at inference, matching STE bitpacked memory while training memory stays higher.
- These numbers come from **scaffolding** runs (few epochs/batches); absolute
  accuracies are not final. The tables validate the analytics pipeline.
