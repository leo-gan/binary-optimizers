from binary_optimizers.benchmarks.cifar10 import run_cifar10_benchmark


def main():
    runs = {
        "STE_SGD (SmallConvNetSTE)": {"model": "small_convnet_ste", "optimizer": "ste_sgd"},
        "Voting (SmallConvNetSTE)": {"model": "small_convnet_ste", "optimizer": "voting"},
        "Signum (SmallBitConvNet)": {"model": "small_bitconvnet", "optimizer": "signum"},
        "Adam (SmallBitConvNet)": {"model": "small_bitconvnet", "optimizer": "adam"},
    }

    run_cifar10_benchmark(runs=runs, epochs=5, seed=42, data_root=".")


if __name__ == "__main__":
    main()
