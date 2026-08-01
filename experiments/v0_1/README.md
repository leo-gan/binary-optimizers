# Experiment v0.1 (`v0_1`)

Latent-free **binary Swarm** training of a small **BitNet-style MLP** on MNIST.

See [PROTOCOL.md](PROTOCOL.md) for claims and design.

## Run

From the repo root:

```bash
# All three LayerNorm modes to early-stop convergence
python experiments/v0_1/train.py --ln-mode all

# Single mode
python experiments/v0_1/train.py --ln-mode no_affine --epochs 80 --patience 10
```

Outputs:

- `results/v0_1/ln_{none,no_affine,affine}_seed42.json`
- `results/v0_1/summary_seed42.json`
- `checkpoints/v0_1/ln_*_seed42.pt`

## Defaults

| Knob | Value |
|------|--------|
| hidden | 128 |
| swarm_size | 32 |
| recruit_rate | 1e4 |
| max_flip_prob | 0.15 |
| grad_momentum | 0.9 |
| activation | relu |
| max epochs | 80 |
| patience | 10 (test acc) |
| min_delta | 5e-4 |
