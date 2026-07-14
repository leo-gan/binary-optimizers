# Inference Benchmark Report

**Device:** cpu  
**Warmup / Iters:** 2 / 5  
**Note:** Scaffolding benchmark — few iterations for pipeline validation.

## Single linear layer (out=64, in=256)

| Mode | Batch | Mean (ms) | Std (ms) | Speedup vs float |
| :--- | :---: | --------: | -------: | ---------------: |
| float | 1 | 0.0036 | 0.0006 | 1.00x |
| sign_weight_float | 1 | 0.0119 | 0.0003 | 0.31x |
| packed_binary | 1 | 0.0993 | 0.0018 | 0.04x |
| float | 8 | 0.0129 | 0.0006 | 1.00x |
| sign_weight_float | 8 | 0.0299 | 0.0017 | 0.43x |
| packed_binary | 8 | 0.2160 | 0.0105 | 0.06x |
| float | 32 | 0.0139 | 0.0003 | 1.00x |
| sign_weight_float | 32 | 0.0641 | 0.0018 | 0.22x |
| packed_binary | 32 | 0.4971 | 0.0506 | 0.03x |

## MNIST Bit-MLP (hidden=64)

### Memory

- Float params: **203776** bytes
- Int8 weights: **50816** bytes
- Bitpacked: **6352** bytes
- Compression float→bitpacked: **32.1x**

### Latency

| Mode | Batch | Mean (ms) | Speedup vs float |
| :--- | :---: | --------: | ---------------: |
| float | 1 | 0.0935 | 1.00x |
| sign_weight_float | 1 | 0.2502 | 0.37x |
| packed_binary | 1 | 0.4118 | 0.23x |
| float | 8 | 0.1356 | 1.00x |
| sign_weight_float | 8 | 0.2830 | 0.48x |
| packed_binary | 8 | 0.5541 | 0.24x |
| float | 32 | 0.1141 | 1.00x |
| sign_weight_float | 32 | 0.2400 | 0.48x |
| packed_binary | 32 | 2.0211 | 0.06x |

## Swarm: full population vs majority-vote cache

- Swarm size: 16, hidden: 64, batch: 32
- Full population forward: **0.3354 ms**
- Majority-cached packed: **0.8824 ms**
- Speedup (majority vs full): **0.38x**
- Bitpacked memory full / majority: **101632** / **6352** bytes (ratio **16.0x**)

## Takeaways

- **Packed binary** replaces float matmul with XNOR+popcount on uint8 bitplanes.
  The reference engine is pure PyTorch (software popcount) for *correctness* and
  API completeness — wall-clock on CPU may lag highly optimized float GEMM.
  Memory compression (float → bitpacked) is the primary measurable win here.
- **Sign-weight float** isolates the arithmetic effect of ±1 weights without packing.
- **Swarm majority cache** collapses population tensors to a single ±1 matrix,
  cutting inference *memory* by ~swarm_size vs carrying the full population.
