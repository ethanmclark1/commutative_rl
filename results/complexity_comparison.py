import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom


def plot_complexity_comparison(action_space_size: int, max_horizon: int) -> None:
    horizons = np.arange(1, max_horizon + 1)

    # Calculate log values directly to avoid overflow
    # Traditional RL complexity: |A|^h = h * log(|A|) in log space
    log_traditional = [h * np.log10(action_space_size) for h in horizons]

    # Commutative RL complexity - use logarithm of binomial coefficient
    log_commutative = []
    for h in horizons:
        # Use logarithm of binomial coefficient for numerical stability
        # log(C(h+|A|-1, |A|-1))
        log_comm = np.log10(binom(h + action_space_size - 1, action_space_size - 1))
        log_commutative.append(log_comm)

    plt.figure(figsize=(10, 6))
    # Plot log10 values directly
    plt.plot(
        horizons, log_traditional, "r-", linewidth=2, label=f"Traditional RL: O(|A|^h)"
    )
    plt.plot(
        horizons,
        log_commutative,
        "b-",
        linewidth=2,
        label=f"Tailored solution: O(h^(|A|-1))",
    )

    # Add explanation for log10 scale
    log_explanation = """Log10 Scale Interpretation:
    • Log10 = 3 → 10³ = 1,000 states
    • Log10 = 6 → 10⁶ = 1,000,000 states
    • Log10 = 9 → 10⁹ = 1 billion states

    Each +1 on y-axis = 10× more states"""

    plt.text(
        0.02,
        0.98,
        log_explanation,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Mark a couple of specific points for reference
    h_examples = [10, 50]
    for h in h_examples:
        if h <= max_horizon:
            trad_val = h * np.log10(action_space_size)
            comm_val = np.log10(binom(h + action_space_size - 1, action_space_size - 1))

            # Add markers
            plt.plot(h, trad_val, "ro", markersize=6)
            plt.plot(h, comm_val, "bo", markersize=6)

            # Add annotations
            plt.annotate(
                f"h={h}: 10^{trad_val:.1f}",
                xy=(h, trad_val),
                xytext=(h + 5, trad_val),
                arrowprops=dict(arrowstyle="->", color="darkred"),
                color="darkred",
                fontsize=8,
            )

            plt.annotate(
                f"h={h}: 10^{comm_val:.1f}",
                xy=(h, comm_val),
                xytext=(h + 5, comm_val),
                arrowprops=dict(arrowstyle="->", color="darkblue"),
                color="darkblue",
                fontsize=8,
            )

    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.title(f"Log₁₀ of State Space Size (|A|={action_space_size})", fontsize=14)
    plt.xlabel("Horizon (h)", fontsize=12)
    plt.ylabel("Log₁₀(Number of States)", fontsize=12)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("complexity_comparison.png", dpi=300)
    plt.show()


plot_complexity_comparison(action_space_size=20, max_horizon=100)
