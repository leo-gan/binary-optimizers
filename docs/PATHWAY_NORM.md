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

**Flip-search rule:** `forward` updates EMAs only when `track=True`. The
v0.13 train loop sets `track=False` during the \(1+k\) trial closures and
calls `update_stats` once on the **accepted** hidden state. Updating on
rejected flips pollutes the motif (see critique below).

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

Scoreboard: [`experiments/v0_13_pathway/RESULTS.md`](../experiments/v0_13_pathway/RESULTS.md).

Motif isolation (3 arms × seeds `{42,0,1}`; pass rule in RESULTS §5):

```bash
python experiments/v0_13_pathway/isolate_motif.py --seeds 42,0,1
```

---

## Critique (after the LN bake-off)

The useful piece is **per-sample L1 scale**. That is why CE dropped from
~45 to ~2. The “assembled motif” is a lagged per-channel L1 mean of
signs, not a reusable building block:

- The same motif is subtracted from every sample. A class-selective
  channel is pushed toward a global firing-rate prototype — the opposite
  of a pathway that depends on the input.
- Agreement froze at ~0.36. Residual L1 never shrank (~20.7). The motif
  does not capture more of \(x\) as training proceeds.
- Subtracting \(\mathrm{sign}(\mathrm{EMA}[s])\cdot\mathrm{EMA}[\lvert x\rvert]\)
  is homeostasis + scale, close to `HomeostaticThreshold`, not assembly.
- **Bug (fixed):** trial flips inside `PathwayOptimizer.step` used to
  update the EMA. Rejected hypotheses wrote the memory. Accept rate rose
  ~0.33 → ~0.40 after updating only the accepted state.

Ablation (`strength=0`, L1 scale only, same tracking fix, same wall)
matched motif-on within noise. Keep the motif as the designed object;
do not claim it is why the net learns.

---

## Follow-up cells (seed 42, 1200 s, `patience_frac=1`)

| Arm | Best test | Best ep | Accept | vs LN 0.6514 |
|-----|----------:|--------:|-------:|-------------:|
| PathwayNorm, EMA on every trial (old) | 0.6368 | 14 | ~0.33 | −1.46 pp |
| LayerNorm, no affine | 0.6514 | 16 | ~0.33 | — |
| **PathwayNorm, EMA on accepted state only** | **0.6750** | 16 | ~0.40 | **+2.36 pp** |
| PathwayNorm, `strength=0` (L1 only) | 0.6732 | 13 | ~0.32 | +2.18 pp |

One seed. Motif vs L1 is a tie. The tracking fix is the real gain.
