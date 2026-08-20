# Pathway optimizer

**Latent-free discrete optimizer:** each 2-D weight is \(\pm 1\). A step
proposes a few bit flips and keeps a flip only if it lowers task loss and/or
a compression-based assembly-index proxy.

This is **not** Swarm. There is no population per weight. Pathway flips
individual binary weights.

Research essay (Assembly Theory motivation, not protocol):
[`docs/temp/pathway_optimizer.md`](temp/pathway_optimizer.md).  
Existence experiment: [`experiments/v0_13_pathway/`](../experiments/v0_13_pathway/).

---

## Assembly-index proxy

Exact Assembly Index is NP-hard. We use a Lempel–Ziv-style ratio on **packed
signs**:

\[
\widehat{A}(t)=\frac{\mathrm{len}(\mathrm{zlib.compress}(\mathrm{packbits}[t \ge 0]))}{\mathrm{len}(\mathrm{packbits}[t \ge 0])}
\]

Random \(\pm 1\) tensors sit near 1. Tiled / repeated patterns are smaller.

**Do not compress raw float32 `±1.0` bytes.** The IEEE payload is shared
across random signs and fakes structure.

`zlib` is **not differentiable**. \(\widehat{A}\) is a metric and a discrete
accept/reject signal. It is not added to an STE loss for backprop.

Implemented in `binary_optimizers/assembly.py`.

---

## Update rule

`PathwayOptimizer` (`binary_optimizers/optimizers/pathway.py`) is
closure-based (LBFGS-style). The closure must re-forward the current batch
and return `(loss, logits)` (or just `loss`).

Defaults: `candidates=8`, `lambda_a=0.1`, `ranking=ste_topk`,
`assembly_of=both`, `mode=pathway`.

Per minibatch:

1. Snap 2-D params to \(\pm 1\) (0 → +1).
2. `loss0, logits0 = closure()`.
3. \(A_0 = 0.5\,\widehat{A}(W)+0.5\,\widehat{A}(\mathrm{logits})\) when
   `assembly_of=both`.
4. Pick `candidates` coordinates among 2-D weights:
   - `ste_topk`: largest \(|\mathrm{grad}|\)
   - `random`: uniform without replacement (top-k of rand)
5. For each candidate, sequentially:
   - flip the bit
   - `loss1, logits1 = closure()`
   - \(\mathrm{score}=\Delta L + \lambda\Delta A\)
   - **accept iff `score < 0`**; else revert
   - on accept, the new state becomes the baseline (greedy path)
6. 1-D params (BN / LN affine) take optional SGD (`bn_lr`); they are never
   flipped.
7. Re-snap 2-D params to \(\pm 1\).

**Target alignment** is the task loss. Do not invent a class-conditional
compressor.

**Cost:** \(1+k\) forwards per step.

### Hybrid mode

Same accept rule. The candidate **pool** is only coordinates where STE wants
a flip: `p * grad > 0` (gradient points the same way as the current sign, so
a descent step would cross zero).

This is a discrete filter on STE-proposed flips, not Adam-on-latent plus a
zlib loss term.

---

## PathwayNorm

The essay’s hierarchical / memory rules live in a **layer**, not in `step()`.
`PathwayNorm` keeps a running assembled \(\pm 1\) motif per channel and
rescales the residual with a per-sample L1. That is the piece the existence
cell was missing: activations, not \(W\), are where \(\pm 1\) matmuls explode
and where structure can actually change every batch.

Write-up: [`PATHWAY_NORM.md`](PATHWAY_NORM.md). Hook: `--norm pathway`.

## What v1 does not do

- Hierarchical “layer 2 may not assemble without layer-1 motifs”
- Memory bank / splice historical weight blocks
- Exact assembly index or grammar induction
- CIFAR / conv nets

---

## Tests and run

```bash
pytest tests/test_pathway.py experiments/v0_13_pathway/test_v0_13_pathway.py -q
python experiments/v0_13_pathway/train.py --seed 42 --max-wall-sec 60
```
