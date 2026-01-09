import argparse
from typing import NoReturn


def _missing_bench_deps() -> NoReturn:
    raise SystemExit(
        "Missing benchmark dependencies. Install with: uv sync --extra bench (or: pip install -e '.[bench]')"
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)


def benchmark_cifar10() -> None:
    try:
        from binary_optimizers.benchmarks.cifar10 import run_cifar10_benchmark
    except ImportError:
        _missing_bench_deps()

    parser = argparse.ArgumentParser(prog="benchmark-cifar10")
    _add_common_args(parser)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    runs = {
        "STE_SGD (SmallConvNetSTE)": {"model": "small_convnet_ste", "optimizer": "ste_sgd"},
        "Voting (SmallConvNetSTE)": {"model": "small_convnet_ste", "optimizer": "voting"},
        "Signum (SmallBitConvNet)": {"model": "small_bitconvnet", "optimizer": "signum"},
        "Adam (SmallBitConvNet)": {"model": "small_bitconvnet", "optimizer": "adam"},
    }

    epochs = args.epochs if args.epochs is not None else 5
    data_root = args.data_root if args.data_root is not None else "."

    run_cifar10_benchmark(
        runs=runs,
        epochs=epochs,
        seed=args.seed,
        device=args.device,
        data_root=data_root,
        num_workers=args.num_workers,
    )


def benchmark_mnist() -> None:
    try:
        from binary_optimizers.benchmarks.mnist import run_mnist_benchmark
    except ImportError:
        _missing_bench_deps()

    parser = argparse.ArgumentParser(prog="benchmark-mnist")
    _add_common_args(parser)
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else 15
    data_root = args.data_root if args.data_root is not None else "./data"

    run_mnist_benchmark(
        epochs=epochs,
        seed=args.seed,
        device=args.device,
        data_root=data_root,
    )
