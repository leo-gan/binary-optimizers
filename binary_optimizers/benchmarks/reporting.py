from typing import Dict, Any


def format_benchmark_results(results: Dict[str, any]) -> str:
    """
    Formats the benchmarking results into a Markdown table.
    """
    lines = []
    lines.append("# Benchmark Results Comparison")
    lines.append("")

    # Get all run names
    run_names = list(results.keys())
    if not run_names:
        return "No results found."

    # Determine number of epochs from the first result
    first_res = results[run_names[0]]
    num_epochs = len(first_res.test_acc_mean)
    trials = first_res.trials

    lines.append(f"**Trials:** {trials} | **Epochs:** {num_epochs}")
    lines.append("")

    # Table header
    header = (
        "| Optimizer | "
        + " | ".join([f"Epoch {i + 1}" for i in range(num_epochs)])
        + " | Mean Time (s) |"
    )
    separator = (
        "| :--- | " + " | ".join([":---:" for _ in range(num_epochs)]) + " | :---: |"
    )
    lines.append(header)
    lines.append(separator)

    for name, res in results.items():
        acc_str = []
        for mean, std in zip(res.test_acc_mean, res.test_acc_std):
            acc_str.append(f"{mean:.4f} ± {std:.4f}")

        avg_time = sum(res.epoch_time_mean) / len(res.epoch_time_mean)

        row = f"| {name} | " + " | ".join(acc_str) + f" | {avg_time:.2f} |"
        lines.append(row)

    return "\n".join(lines)
