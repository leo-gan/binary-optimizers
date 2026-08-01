# STE vs Swarm

Shared-protocol MNIST comparison: STE binary MLP vs Swarm v0.3 / v0.4.

See [PROTOCOL.md](PROTOCOL.md).

```bash
# Full grid (3 methods × 3 LN modes) — long on CPU
python experiments/ste_vs_swarm/train.py

# Faster subset
python experiments/ste_vs_swarm/train.py --methods ste_sgd,swarm_v0_3 --ln-mode affine --epochs 5

# Full store report (includes STE vs Swarm section)
./scripts/report.sh

# STE vs Swarm comparison report only
./scripts/report_ste_vs_swarm.sh
./scripts/report_ste_vs_swarm.sh > results/ste_vs_swarm_report.md
```
