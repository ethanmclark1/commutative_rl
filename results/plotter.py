import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

df["Step"] = df["Step"] / 1_000_000

methods = {
    "Traditional": {"color": "red", "label": "Traditional Q-Table"},
    "Commutative": {"color": "green", "label": "Commutative Q-Table"},
    "TripleTraditional": {"color": "blue", "label": "Triple Traditional Q-Table"},
}

plt.figure(figsize=(6, 4))

for method, props in methods.items():
    mean = df[f"{method} DQN - Average Return"]
    max_val = df[f"{method} DQN - Average Return__MAX"]
    min_val = df[f"{method} DQN - Average Return__MIN"]

    std_dev = (max_val - min_val) / 4

    plt.plot(df["Step"], mean, label=props["label"], color=props["color"], linewidth=2)
    plt.fill_between(
        df["Step"], mean - std_dev, mean + std_dev, color=props["color"], alpha=0.2
    )

plt.title("Average Return", fontsize=12, pad=15)
plt.xlabel("Step (1e6)", fontsize=10)
plt.legend(fontsize=9)

plt.xticks(range(0, 6))
plt.ylim(bottom=0)

plt.tight_layout()

plt.savefig("convergence.png", dpi=300, bbox_inches="tight")
plt.close()
