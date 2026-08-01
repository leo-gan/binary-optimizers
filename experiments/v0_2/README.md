# Experiment v0.2 (`v0_2`)

Place-value (**exponential**) binary Swarm — extension of [v0.1](../v0_1/).

See [PROTOCOL.md](PROTOCOL.md).

## Run

```bash
# All LayerNorm modes
python experiments/v0_2/train.py --ln-mode all

# Compare to v0.1 defaults: n_bits=16 vs unary swarm_size=32
python experiments/v0_2/train.py --ln-mode none --n-bits 16
```

Outputs: `results/v0_2/`, `checkpoints/v0_2/`.

## Defaults

| Knob | Value |
|------|--------|
| n_bits | 16 (place \(2^i\)) |
| coding | place-value, binary forward |
| lsb_bias | on |
| hidden | 128 |
| recruit_rate | 1e4 |
| max_flip_prob | 0.15 |
| grad_momentum | 0.9 |
