#!/usr/bin/env python3
"""Download MNIST and/or CIFAR-10 for local experiments.

Datasets are **not** committed to git. CI/unit tests do not need them.
Run this once after a local clone if you want to train or benchmark.

Examples
--------
  # Both datasets into ./data (default)
  python scripts/download_datasets.py

  # MNIST only
  python scripts/download_datasets.py --mnist --no-cifar

  # Custom root (must match --data-root used by train scripts)
  python scripts/download_datasets.py --data-root ./data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _require_torchvision() -> None:
    try:
        import torchvision  # noqa: F401
    except ImportError:
        print(
            "torchvision is required to download datasets.\n"
            "  uv sync --extra bench\n"
            "  # or: pip install -e '.[bench]'",
            file=sys.stderr,
        )
        raise SystemExit(1)


def download_mnist(root: Path) -> None:
    import torchvision.datasets as datasets
    import torchvision.transforms as T

    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MNIST → {root.resolve()} ...")
    transform = T.ToTensor()
    datasets.MNIST(str(root), train=True, download=True, transform=transform)
    datasets.MNIST(str(root), train=False, download=True, transform=transform)
    print("  MNIST ready.")


def download_cifar10(root: Path) -> None:
    import torchvision.datasets as datasets
    import torchvision.transforms as T

    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CIFAR-10 → {root.resolve()} ...")
    transform = T.ToTensor()
    datasets.CIFAR10(str(root), train=True, download=True, transform=transform)
    datasets.CIFAR10(str(root), train=False, download=True, transform=transform)
    print("  CIFAR-10 ready.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MNIST / CIFAR-10 for local binary-optimizers runs"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory for torchvision datasets (default: ./data)",
    )
    parser.add_argument(
        "--mnist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download MNIST (default: yes)",
    )
    parser.add_argument(
        "--cifar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download CIFAR-10 (default: yes)",
    )
    args = parser.parse_args()

    if not args.mnist and not args.cifar:
        print("Nothing to do (both --no-mnist and --no-cifar).", file=sys.stderr)
        raise SystemExit(2)

    _require_torchvision()
    root: Path = args.data_root

    if args.mnist:
        download_mnist(root)
    if args.cifar:
        download_cifar10(root)

    print(f"\nDone. Point train scripts at: --data-root {root}")
    print("Note: data/ is gitignored; re-run this script on a fresh machine.")


if __name__ == "__main__":
    main()
