# Experiment v0.13 — Pathway optimizer (assembly-guided bit flips)

**Path:** `experiments/v0_13_pathway/` · **Results:** `results/v0_13_pathway/`  
**Algorithm:** [`docs/PATHWAY_OPTIMIZER.md`](../../docs/PATHWAY_OPTIMIZER.md)  
**Essay (not protocol):** [`docs/temp/pathway_optimizer.md`](../../docs/temp/pathway_optimizer.md)

## Claim

Train a **latent-free** binary MLP on **MNIST** by flipping individual ±1
weights. A flip is kept only when it lowers

\[
\mathrm{score} = \Delta L + \lambda\,\Delta\widehat{A}
\]

where \(L\) is minibatch cross-entropy and \(\widehat{A}\) is the **zlib
packed-sign** assembly-index proxy (not Cronin’s exact \(A\)).

This is a **new discrete-optimizer line**, not a Swarm encoding atlas.
There is no population / swarm dimension.

## Non-claims

- Not Swarm, not place-value, not unary XOR writeback.
- Not exact Assembly Index, grammar induction, or a differentiable AT loss
  (`zlib` cannot backprop).
- Not hierarchical layer-wise assembly discipline.
- Not a memory-bank / splice-from-history optimizer.
- Not CIFAR / conv.
- Activations and autograd remain FP. STE grads are used only to **rank**
  candidate coordinates (`ste_topk`) or to define the hybrid pool.

## Architecture

```
Flatten
→ BitLinearSTE(784 → H)   # snapped to ±1 at init; stays ±1
→ ReLU
→ BitLinearSTE(H → 10)
→ CrossEntropy
```

Defaults: `H=128`, no bias, **`--norm none`** in the existence cell.
`--norm pathway` inserts [`PathwayNorm`](../../docs/PATHWAY_NORM.md) on the
hidden pre-activations (assembled ±1 motif + L1 residual scale).
`--norm layernorm` / `--norm bn` (`--bn`) are FP crutches.

## Optimizer

`PathwayOptimizer` (`binary_optimizers/optimizers/pathway.py`).

| Item | Default |
|------|---------|
| Mode | `pathway` (also `hybrid`) |
| Candidates \(k\) | 8 sequential trial flips per step |
| Ranking | `ste_topk` (largest \(\lvert g\rvert\)); `random` allowed |
| \(\lambda\) (`lambda_a`) | 0.1 |
| \(\widehat{A}\) over | `both` = \(0.5\,\widehat{A}(W)+0.5\,\widehat{A}(\mathrm{logits})\) |
| Accept | `score < 0` (greedy; accepted state becomes the baseline) |
| 1-D params | not flipped; optional SGD `bn_lr=1e-2` |

**`hybrid`:** same accept rule, but the candidate **pool** is only
coordinates where STE wants a flip (`p * grad > 0`).

**Cost:** \(1+k\) forwards per minibatch.

## \(\widehat{A}\) definition

Pack signs (`>=0` → 1) with `numpy.packbits`, then

\[
\widehat{A}(t)=\frac{\mathrm{len}(\mathrm{zlib.compress}(\mathrm{pack}(t)))}{\mathrm{len}(\mathrm{pack}(t))}.
\]

Packed bits, not raw float32 bytes (IEEE `±1.0` compresses even when signs
are random).

## Success criteria

- Test accuracy rises above chance and plateaus under `pure_wall_budget_v1`.
- After every step: linear weights \(\in\{+1,-1\}\).
- Log \(\widehat{A}(W)\) and accept/propose ratio.

v1 is an **existence** cell, not a claim that pathway beats Swarm / STE.

## Results (seed=42, defaults, `pure_wall_budget_v1`)

| Arm | best test acc | best epoch | epochs ran | stop | wall |
|-----|---------------|------------|------------|------|------|
| `pathway` smoke (`max_wall_sec=60`) | 0.5158 | 1 | 1 | max_wall_sec | 77 s |
| `pathway` existence | **0.6103** | 11 | 13 | patience_wall | 1035 s |
| `--norm pathway` smoke | 0.5573 | 1 | 2 | max_wall_sec | 119 s |
| `--norm pathway` existence | **0.5573** | 1 | 4 | patience_wall | 230 s |

Accept rate stayed ~0.28–0.32. \(\widehat{A}(W)\) stayed ≈1.00 (too few flips
to structure the full matrix). Chance is 0.10; existence holds. Not a
comparison to Swarm v0.1 LN0 (0.9239).

Artifacts: `results/v0_13_pathway/`, `checkpoints/v0_13_pathway/`.

## Norm comparison (PathwayNorm vs LayerNorm)

**Question:** under the same flip optimizer and wall, does PathwayNorm beat
or match a standard hidden norm?

| Item | Value |
|------|--------|
| Arms | `--norm pathway` vs `--norm layernorm` (`LayerNorm`, no affine) |
| Optimizer | same `PathwayOptimizer` defaults |
| Seed | 42 |
| Budget | `pure_wall_budget_v1`, **patience = full wall** (`patience_frac=1`) so a test stall cannot cut one arm short |
| Standard | LayerNorm is the repo’s usual MLP norm (v0.1 LN1). Not BatchNorm. |

```bash
python experiments/v0_13_pathway/compare_norms.py --seed 42
```

Writes `results/v0_13_pathway/compare_norms_seed42.{json,md}` plus per-arm
`cmp_<norm>_seed42.json`.

### Results (seed=42, 1200 s wall, `patience_frac=1`)

| norm | best test | best ep | epochs | wall s | final loss |
|------|----------:|--------:|-------:|-------:|-----------:|
| `pathway` (PathwayNorm) | 0.6368 | 14 | 18 | 1241 | 2.220 |
| `layernorm` (LN, no affine) | **0.6514** | 16 | 17 | 1259 | **1.727** |

LayerNorm wins by 1.46 pp. Both beat the no-norm existence cell (0.6103).
One seed; not a Swarm comparison.

**Superseded for PathwayNorm:** that 0.6368 cell updated the motif EMA on
every trial flip (including rejects). After fixing that:

| Arm | Best test | Best ep | Accept |
|-----|----------:|--------:|-------:|
| PathwayNorm, EMA on accepted state | **0.6750** | 16 | ~0.40 |
| PathwayNorm, `strength=0` (L1 only) | 0.6732 | 13 | ~0.32 |
| LayerNorm (table above) | 0.6514 | 16 | ~0.33 |

Tags: `cmp_pathway_fix`, `cmp_pathway_l1`. Motif vs L1 is a tie; the
tracking fix is the gain vs LN.

## Scoreboard

Consolidated numbers and the pre-registered motif-isolation pass rule:
[`RESULTS.md`](RESULTS.md). Isolation **FAIL** (means: LN 0.6682, L1 0.6664,
motif 0.6668).

## Defaults (existence)

| Item | Value |
|------|--------|
| Data | MNIST |
| Seed | 42 |
| Batch | 128 |
| Budget | `pure_wall_budget_v1` (1200 s wall) |
| Mode | `pathway` |

## Run

```bash
pytest experiments/v0_13_pathway/test_v0_13_pathway.py tests/test_pathway.py -q

python experiments/v0_13_pathway/train.py --seed 42 --max-wall-sec 60
python experiments/v0_13_pathway/train.py --seed 42
python experiments/v0_13_pathway/train.py --norm pathway --run-tag pathway_norm
python experiments/v0_13_pathway/train.py --mode hybrid --run-tag hybrid
```
