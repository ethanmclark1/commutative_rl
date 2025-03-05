import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

df["Step"] = df["Step"] / 1_000_000

# Modified method names and labels
methods = {
    "Name: TraditionalDQN": {"color": "red", "label": "DQN"},
    "Name: TripleTraditionalDQN": {"color": "blue", "label": "Triple Data DQN"},
    "Name: CommutativeDQN": {"color": "green", "label": "Commutative DQN"},
}

plt.figure(figsize=(6, 4))

for method, props in methods.items():
    mean = df[f"{method} - Average Return"]
    max_val = df[f"{method} - Average Return__MAX"]
    min_val = df[f"{method} - Average Return__MIN"]

    std_dev = (max_val - min_val) / 4

    plt.plot(df["Step"], mean, label=props["label"], color=props["color"], linewidth=2)
    plt.fill_between(
        df["Step"], mean - std_dev, mean + std_dev, color=props["color"], alpha=0.2
    )

max_step = round(df["Step"].max()) + 1

plt.title("Average Return", fontsize=12, pad=15)
plt.xlabel(f"Step (1e{max_step})", fontsize=10)
plt.legend(fontsize=9)

plt.xticks(range(0, max_step))
plt.ylim(bottom=0)

plt.tight_layout()

plt.savefig("convergence.png", dpi=300, bbox_inches="tight")
plt.close()
