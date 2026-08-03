# v0.6 encoding atlas

```bash
# Primary grid: n∈{8,16} × encodings + unary S=256 baseline
python experiments/v0_6_encoding/train.py --ln-mode none --seed 42

# Include n=32 rescue probe
python experiments/v0_6_encoding/train.py --include-rescue

# Subset
python experiments/v0_6_encoding/train.py --cells fixed@8,exp_mant:2@8,unary:256
```

Results: `results/v0_6_encoding/`. See [PROTOCOL.md](PROTOCOL.md) and Plan WP2.
