# Pareto Analysis Report

Combines scaffolding training-sweep results with inference benchmarks.

**Note:** Scaffolding run — short epochs/batches to demonstrate pipeline.

## Full comparison

| Config | Dataset | Test Acc (mean ± std) | Infer mem (bitpacked) | Infer latency (ms) | Train mem | Train time (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| mnist_small_threshold_if | mnist | 0.3086 ± 0.0829 | 6.4 KB | 0.4308 | 407.0 KB | 0.11 |
| mnist_swarm | mnist | 0.2695 ± 0.0387 | 6.4 KB | 0.8973 | 1.63 MB | 0.08 |
| mnist_small_voting | mnist | 0.2246 ± 0.0470 | 6.4 KB | 0.4308 | 407.0 KB | 0.06 |
| mnist_small_ste | mnist | 0.2168 ± 0.1464 | 6.4 KB | 0.4308 | 407.6 KB | 0.06 |
| mnist_large_ste | mnist | 0.1992 ± 0.0552 | 14.8 KB | 0.4308 | 948.2 KB | 0.11 |
| mnist_large_threshold_if | mnist | 0.1992 ± 0.0442 | 14.8 KB | 0.4308 | 946.2 KB | 0.07 |
| mnist_large_voting | mnist | 0.1953 ± 0.0055 | 14.8 KB | 0.4308 | 946.2 KB | 0.09 |
| mnist_small_adam | mnist | 0.1426 ± 0.0967 | 6.4 KB | 0.4308 | 611.3 KB | 0.07 |
| mnist_small_signum | mnist | 0.1426 ± 0.0967 | 6.4 KB | 0.4308 | 407.0 KB | 0.06 |
| cifar_adam | cifar10 | 0.1348 ± 0.0083 | 262.5 KB | 0.1151 | 26.11 MB | 0.49 |
| cifar_signum | cifar10 | 0.1328 ± 0.0000 | 262.5 KB | 0.1151 | 17.40 MB | 0.53 |
| cifar_ste_sgd | cifar10 | 0.1172 ± 0.0000 | 131.4 KB | 0.1151 | 8.57 MB | 0.36 |
| mnist_large_signum | mnist | 0.0723 ± 0.0028 | 14.8 KB | 0.4308 | 946.2 KB | 0.06 |
| mnist_large_adam | mnist | 0.0703 ± 0.0055 | 14.8 KB | 0.4308 | 1.42 MB | 0.07 |

## Accuracy vs inference memory

| Config | Acc | Bitpacked mem |
| :--- | ---: | ---: |
| mnist_small_adam | 0.1426 | 6.4 KB |
| mnist_small_ste | 0.2168 | 6.4 KB |
| mnist_small_voting | 0.2246 | 6.4 KB |
| mnist_small_signum | 0.1426 | 6.4 KB |
| mnist_small_threshold_if ★ | 0.3086 | 6.4 KB |
| mnist_swarm | 0.2695 | 6.4 KB |
| mnist_large_adam | 0.0703 | 14.8 KB |
| mnist_large_ste | 0.1992 | 14.8 KB |
| mnist_large_voting | 0.1953 | 14.8 KB |
| mnist_large_signum | 0.0723 | 14.8 KB |
| mnist_large_threshold_if | 0.1992 | 14.8 KB |
| cifar_ste_sgd | 0.1172 | 131.4 KB |
| cifar_adam | 0.1348 | 262.5 KB |
| cifar_signum | 0.1328 | 262.5 KB |

## Accuracy vs inference speed

| Config | Acc | Latency (ms) |
| :--- | ---: | ---: |
| cifar_adam ★ | 0.1348 | 0.1151 |
| cifar_signum | 0.1328 | 0.1151 |
| cifar_ste_sgd | 0.1172 | 0.1151 |
| mnist_small_adam | 0.1426 | 0.4308 |
| mnist_small_ste | 0.2168 | 0.4308 |
| mnist_small_voting | 0.2246 | 0.4308 |
| mnist_small_signum | 0.1426 | 0.4308 |
| mnist_small_threshold_if ★ | 0.3086 | 0.4308 |
| mnist_large_adam | 0.0703 | 0.4308 |
| mnist_large_ste | 0.1992 | 0.4308 |
| mnist_large_voting | 0.1953 | 0.4308 |
| mnist_large_signum | 0.0723 | 0.4308 |
| mnist_large_threshold_if | 0.1992 | 0.4308 |
| mnist_swarm | 0.2695 | 0.8973 |

## Training efficiency (acc vs train mem & time)

| Config | Acc | Train mem | Time (s) |
| :--- | ---: | ---: | ---: |
| mnist_small_signum ★ | 0.1426 | 407.0 KB | 0.06 |
| mnist_small_ste ★ | 0.2168 | 407.6 KB | 0.06 |
| mnist_small_voting ★ | 0.2246 | 407.0 KB | 0.06 |
| mnist_large_signum | 0.0723 | 946.2 KB | 0.06 |
| mnist_large_adam | 0.0703 | 1.42 MB | 0.07 |
| mnist_large_threshold_if | 0.1992 | 946.2 KB | 0.07 |
| mnist_small_adam | 0.1426 | 611.3 KB | 0.07 |
| mnist_swarm ★ | 0.2695 | 1.63 MB | 0.08 |
| mnist_large_voting | 0.1953 | 946.2 KB | 0.09 |
| mnist_large_ste | 0.1992 | 948.2 KB | 0.11 |
| mnist_small_threshold_if ★ | 0.3086 | 407.0 KB | 0.11 |
| cifar_ste_sgd | 0.1172 | 8.57 MB | 0.36 |
| cifar_adam | 0.1348 | 26.11 MB | 0.49 |
| cifar_signum | 0.1328 | 17.40 MB | 0.53 |

## Pareto-optimal configurations

- **Accuracy vs inference memory:** mnist_small_threshold_if
- **Accuracy vs inference speed:** mnist_small_threshold_if, cifar_adam
- **Training efficiency:** mnist_small_ste, mnist_small_voting, mnist_small_signum, mnist_small_threshold_if, mnist_swarm
- **All objectives:** mnist_small_ste, mnist_small_voting, mnist_small_signum, mnist_small_threshold_if, mnist_swarm, cifar_adam, cifar_signum, cifar_ste_sgd

★ marks Pareto-optimal points on the corresponding frontier.

## Interpretation

- Higher accuracy with lower bitpacked memory is preferred for edge deployment.
- Swarm configs store a population during training but can collapse to a majority
  vote at inference, matching STE bitpacked memory while training memory stays higher.
- These numbers come from **scaffolding** runs (few epochs/batches); absolute
  accuracies are not final. The tables validate the analytics pipeline.
