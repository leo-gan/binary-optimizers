# v0.13 — Pathway optimizer

Assembly-guided ±1 flip search on a latent-free binary MLP.

Protocol: [`PROTOCOL.md`](PROTOCOL.md)  
Library: `binary_optimizers/optimizers/pathway.py`, `binary_optimizers/assembly.py`  
Write-up: [`docs/PATHWAY_OPTIMIZER.md`](../../docs/PATHWAY_OPTIMIZER.md)

## Quick start

```bash
# Unit tests (no dataset)
pytest experiments/v0_13_pathway/test_v0_13_pathway.py tests/test_pathway.py -q

# Smoke (1 minute wall)
python experiments/v0_13_pathway/train.py --seed 42 --max-wall-sec 60 --run-tag smoke

# Existence cell (default 20 min wall)
python experiments/v0_13_pathway/train.py --seed 42 --run-tag pathway

# Hybrid STE-pool + same accept rule
python experiments/v0_13_pathway/train.py --mode hybrid --seed 42 --run-tag hybrid
```

`--lambda-a 0` is a legal ablation (loss-only flips).

Hidden-activation norm (after the first binary linear):

```bash
python experiments/v0_13_pathway/train.py --norm pathway --run-tag pathway_norm --seed 42
```

See [`docs/PATHWAY_NORM.md`](../../docs/PATHWAY_NORM.md).

Matched-wall **PathwayNorm vs LayerNorm** (patience = full wall):

```bash
python experiments/v0_13_pathway/compare_norms.py --seed 42
```

PathwayNorm motif ablation (L1 scale only) and the accepted-state EMA
update are in `train.py` (`--norm-strength 0`; `track=False` during flip
closures). See [`docs/PATHWAY_NORM.md`](../../docs/PATHWAY_NORM.md).
