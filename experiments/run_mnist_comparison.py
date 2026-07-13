from binary_optimizers.benchmarks.mnist import run_mnist_benchmark
from binary_optimizers.benchmarks.reporting import format_benchmark_results


def main():
    print("Starting MNIST Comparison Experiment...")
    results = run_mnist_benchmark(epochs=5, num_trials=3, seed=42)

    report = format_benchmark_results(results)

    with open("mnist_improved_report.md", "w") as f:
        f.write(report)

    print("\nExperiment Complete. Report saved to mnist_improved_report.md")
    print(report)


if __name__ == "__main__":
    main()
