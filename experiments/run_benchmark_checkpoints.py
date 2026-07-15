#!/usr/bin/env python3
"""Re-evaluate saved checkpoints without training."""

from __future__ import annotations

import argparse
from pathlib import Path

from binary_optimizers.benchmarks.fit_training import (
    FIT_OPTIMIZERS,
    benchmark_checkpoints,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--model", default="bit_mlp", choices=("bit_mlp", "bit_mlp_large"))
    p.add_argument("--optimizers", nargs="*", default=None)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoint-root", default="checkpoints")
    p.add_argument("--output", default="results/checkpoint_benchmark.json")
    args = p.parse_args()

    payload = benchmark_checkpoints(
        optimizers=args.optimizers or FIT_OPTIMIZERS,
        model_name=args.model,
        epochs=args.epochs,
        seed=args.seed,
        batch_size_train=args.batch_size,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        output_json=args.output,
    )
    # Markdown summary
    lines = [
        "# Checkpoint Benchmark (no training)",
        "",
        f"Model `{args.model}`, epochs={args.epochs}, seed={args.seed}",
        "",
        "| Optimizer | STE test | Packed test | Cached best | Status |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]
    for r in payload["rows"]:
        if r["status"] != "ok":
            lines.append(f"| `{r['optimizer']}` | — | — | — | {r['status']} |")
            continue
        lines.append(
            f"| `{r['optimizer']}` | {r['ste_test_acc']:.4f} | {r['packed_test_acc']:.4f} | "
            f"{r.get('cached_best_test_acc')} | ok |"
        )
    md_path = Path("results/checkpoint_benchmark.md")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
