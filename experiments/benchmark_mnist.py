from binary_optimizers.benchmarks.mnist import run_mnist_benchmark


def main():
    run_mnist_benchmark(epochs=15, seed=42, data_root="./data")


if __name__ == "__main__":
    main()
