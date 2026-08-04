# v0.7 CIFAR encoding probe (max learning / min runs)

```bash
# Option B pure wall: 40 min/cell, 300 s stall
python experiments/v0_7_cifar_encoding/train.py --ln-mode none --seed 42
```

Four cells: `fixed@8`, `fixed@16`, `exp_mant:2@8`, `exp_mant:2@16`.

See [PROTOCOL.md](PROTOCOL.md).
