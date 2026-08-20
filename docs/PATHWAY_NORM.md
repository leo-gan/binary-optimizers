# PathwayNorm

A drop-in replacement for LayerNorm / BatchNorm that uses the same
**pathway / assembly** objects as the flip optimizer: a discrete assembled
motif, memory of past signs, and an integer-friendly residual scale.

Essay (motivation, not protocol): [`temp/pathway_optimizer.md`](temp/pathway_optimizer.md)
§ Hierarchical Assembly Constraints and § Memory-Driven Selection.  
Optimizer: [`PATHWAY_OPTIMIZER.md`](PATHWAY_OPTIMIZER.md).  
Code: `binary_optimizers/models/pathway_norm.py`.

This is **not** Swarm homeostasis (`HomeostaticThreshold`) and **not** FP
LayerNorm. The running state is a \(\pm 1\) prototype per channel, not
\(\mu,\sigma\) and not a learned firing-rate threshold.

---

## Why a layer, not another optimizer knob

The v0.13 existence cell showed two things:

1. Greedy \(\pm 1\) flips learn (10% → 61% MNIST).
2. \(\widehat{A}(W)\) (zlib on packed weights) never moved — it cannot
   stabilize the **activations**, which is where \(\pm 1\) matmuls explode
   (unscaled fan-in, CE tens of nats).

The essay’s hierarchical rule is a **forward** constraint: each layer should
see features already assembled below it. That belongs in a module, not in
`step()`. zlib stays a metric; the layer uses a **differentiable / cheap**
stand-in that actually changes every batch.

---

## Mapping (assembly → tensors)

| Assembly idea | In `PathwayNorm` |
|---------------|------------------|
| Building block | Per-channel motif direction \(\in\{+1,-1\}\) |
| Object being assembled | The activation vector (or map) |
| Memory of past steps | EMA of signs and of mean \(\lvert x\rvert\) (buffers, not parameters) |
| Residual / not-yet-assembled | \(x - \mathrm{strength}\cdot m_{\mathrm{dir}}\cdot m_{\mathrm{mag}}\) |
| Complexity / scale | Per-sample L1 of the residual (popcount-like; not \(\sigma\)) |
| Hierarchical discipline | Stack one norm after each binary linear; lower layers form motifs first |

No affine \(\gamma,\beta\) by default (that is the LN2 crutch). Optional
`binarize` applies STE \(\mathrm{sign}\) after the scale (binary activation
path; skip ReLU if you use it).

---

## Forward

Input \(x\) with channel axis 1, shape `[N, C]` or `[N, C, …]`.

**Train** (no grad on buffers):

\[
s=\mathrm{sign}(x),\quad
\bar s_c=\mathbb{E}_{\mathrm{batch,spatial}}[s]_{\,c},\quad
\bar a_c=\mathbb{E}_{\mathrm{batch,spatial}}[\lvert x\rvert]_{\,c}
\]

\[
m^{\mathrm{acc}}\leftarrow (1-\mu)\,m^{\mathrm{acc}}+\mu\,\bar s,\qquad
a^{\mathrm{acc}}\leftarrow (1-\mu)\,a^{\mathrm{acc}}+\mu\,\bar a
\]

**Always:**

\[
m_{\mathrm{dir}}=\mathrm{sign}(m^{\mathrm{acc}}),\quad
m_{\mathrm{mag}}=a^{\mathrm{acc}},\quad
r=x-\alpha\,m_{\mathrm{dir}}\,m_{\mathrm{mag}}
\]

\[
\sigma(x)=\mathrm{mean}_{\mathrm{features}}(\lvert r\rvert)+\varepsilon,\qquad
y=r/\sigma(x)
\]

\(\alpha\) is `strength` (default 1). Scale is **per sample** (eval-safe at
batch 1). Motif is **per channel** (the assembled feature).

Eval uses the frozen EMAs (BatchNorm-style memory, LayerNorm-style scale).

---

## What this is not

- Not \(\mathcal{L}+\lambda A\) in autograd (zlib still is not a backward term).
- Not exact Assembly Index.
- Not the deferred optimizer memory-bank / weight splice.
- Not a claim that PathwayNorm beats LayerNorm on accuracy.

---

## Experiment hook

`experiments/v0_13_pathway/` `--norm {none,pathway,layernorm,bn}`.

Default stays `none` (the existence cell). `--norm pathway` is the new arm:
same flip optimizer, activations centered on the running motif.

Success for this arm: hidden activations O(1), train CE on a normal scale,
test acc at least as stable as `none` (less epoch-12 collapse). Not a
bake-off vs Swarm.

**Head-to-head vs standard LayerNorm** (no affine), same optimizer and
full-wall patience:

```bash
python experiments/v0_13_pathway/compare_norms.py --seed 42
```
