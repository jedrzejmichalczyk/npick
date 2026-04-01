import csv
import sys

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: py -3 examples/plot_matching_response.py <input.csv> <output.png>")
        return 1

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    freq = []
    load_db = []
    g11_opt_db = []
    g11_base_db = []

    with open(input_path, newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            freq.append(float(row["freq"]))
            load_db.append(float(row["load_db"]))
            g11_opt_db.append(float(row["g11_optimized_db"]))
            g11_base_db.append(float(row["g11_baseline_db"]))

    plt.figure(figsize=(10, 5.5))
    plt.plot(freq, load_db, label="Load reflection |Gamma_L|", linewidth=2.0, color="#4c78a8")
    plt.plot(freq, g11_opt_db, label="Matched reflection |Gamma_in|", linewidth=2.2, color="#d95f02")
    plt.plot(freq, g11_base_db, label="Chebyshev nodes", linewidth=1.6, linestyle="--", color="#7f7f7f")
    plt.axvline(-1.0, color="#cccccc", linewidth=1.0)
    plt.axvline(1.0, color="#cccccc", linewidth=1.0)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Normalized frequency")
    plt.ylabel("Reflection magnitude [dB]")
    plt.title("Original vs matched reflection for the first benchmark")
    plt.legend()
    plt.ylim(-15.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
