# Inference Benchmark Report

**Device:** cpu  
**Warmup / Iters:** 2 / 5  
**Note:** Scaffolding benchmark — few iterations for pipeline validation.

## Single linear layer (out=64, in=256)

| Mode | Batch | Mean (ms) | Std (ms) | Speedup vs float |
| :--- | :---: | --------: | -------: | ---------------: |
| float | 1 | 0.0043 | 0.0009 | 1.00x |
| sign_weight_float | 1 | 0.0135 | 0.0005 | 0.32x |
| packed_binary | 1 | 0.1151 | 0.0105 | 0.04x |
| float | 8 | 0.0150 | 0.0003 | 1.00x |
| sign_weight_float | 8 | 0.0396 | 0.0027 | 0.38x |
| packed_binary | 8 | 0.2866 | 0.0256 | 0.05x |
| float | 32 | 0.0187 | 0.0003 | 1.00x |
| sign_weight_float | 32 | 0.0788 | 0.0012 | 0.24x |
| packed_binary | 32 | 0.5560 | 0.2028 | 0.03x |

## MNIST Bit-MLP (hidden=64)

### Memory

- Float params: **203776** bytes
- Int8 weights: **50816** bytes
- Bitpacked: **6352** bytes
- Compression float→bitpacked: **32.1x**

### Latency

| Mode | Batch | Mean (ms) | Speedup vs float |
| :--- | :---: | --------: | ---------------: |
| float | 1 | 0.1038 | 1.00x |
| sign_weight_float | 1 | 0.2776 | 0.37x |
| packed_binary | 1 | 0.4308 | 0.24x |
| float | 8 | 0.1372 | 1.00x |
| sign_weight_float | 8 | 0.2782 | 0.49x |
| packed_binary | 8 | 0.6734 | 0.20x |
| float | 32 | 0.1198 | 1.00x |
| sign_weight_float | 32 | 0.2530 | 0.47x |
| packed_binary | 32 | 1.6934 | 0.07x |

## Swarm: full population vs majority-vote cache

- Swarm size: 16, hidden: 64, batch: 32
- Full population forward: **0.3518 ms**
- Majority-cached packed: **0.8973 ms**
- Speedup (majority vs full): **0.39x**
- Bitpacked memory full / majority: **101632** / **6352** bytes (ratio **16.0x**)

## Takeaways

- **Packed binary** replaces float matmul with XNOR+popcount on uint8 bitplanes.
  The reference engine is pure PyTorch (software popcount) for *correctness* and
  API completeness — wall-clock on CPU may lag highly optimized float GEMM.
  Memory compression (float → bitpacked) is the primary measurable win here.
- **Sign-weight float** isolates the arithmetic effect of ±1 weights without packing.
- **Swarm majority cache** collapses population tensors to a single ±1 matrix,
  cutting inference *memory* by ~swarm_size vs carrying the full population.
