import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

df["Step"] = df["Step"] / 1_000_000

methods = {
    # "Name: HashMapQTable": {
    #     "color": "orange",
    #     "label": "Commutative Q-table (Hash Map)",
    # },
    # "Name: CombinedRewardQTable": {
    #     "color": "brown",
    #     "label": "Commutative Q-table (Combined Reward)",
    # },
    # "Name: SuperActionQTable": {
    #     "color": "red",
    #     "label": "Commutative Q-table (Super Action)",
    # },
    # "Name: TripleDataQTable": {"color": "blue", "label": "Triple Data Q-table"},
    # "Name: QTable": {"color": "green", "label": "Q-table"},
    "Name: HashMapDQN": {"color": "orange", "label": "Commutative DQN (Hash Map)"},
    "Name: CombinedRewardDQN": {
        "color": "brown",
        "label": "Commutative DQN (Combined Reward)",
    },
    "Name: SuperActionDQN": {"color": "red", "label": "Commutative DQN (Super Action)"},
    "Name: TripleDataDQN": {"color": "blue", "label": "Triple Data DQN"},
    "Name: DQN": {"color": "green", "label": "DQN"},
}

plt.figure(figsize=(6, 4))

for method, props in methods.items():
    mean = df[f"{method} - Average Return"]
    max_val = df[f"{method} - Average Return__MAX"]
    min_val = df[f"{method} - Average Return__MIN"]

    std_dev = (max_val - min_val) / 4

    plt.plot(
        df["Step"], mean, label=props["label"], color=props["color"], linewidth=1.5
    )
    plt.fill_between(
        df["Step"], mean - std_dev, mean + std_dev, color=props["color"], alpha=0.2
    )

max_step = round(df["Step"].max())

plt.xlabel("Step (1e6)", fontsize=10)
plt.legend(fontsize=9, loc="lower right")

# plt.xticks(range(0, max_step, 5))
plt.xticks(range(0, max_step, 1))
plt.ylim(bottom=2500, top=5200)

plt.tight_layout()

plt.savefig("convergence.png", dpi=300, bbox_inches="tight")
plt.close()
