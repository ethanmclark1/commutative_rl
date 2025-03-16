import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

df = pd.read_csv("results.csv")

# Instead of skipping rows, we'll preserve the original data but set a consistent x-axis starting point
min_step = df["Step"].min()
df["Step"] = (df["Step"] - min_step) / 1_000_000

# Modified method names and labels
methods = {
    "Name: HashMapQTable": {
        "color": "orange",
        "label": "Commutative Q-table (Hash Map)",
    },
    "Name: CombinedRewardQTable": {
        "color": "brown",
        "label": "Commutative Q-table (Combined Reward)",
    },
    "Name: SuperActionQTable": {
        "color": "red",
        "label": "Commutative Q-table (Super Action)",
    },
    "Name: TripleDataQTable": {"color": "blue", "label": "Triple Data Q-table"},
    "Name: QTable": {"color": "green", "label": "Q-table"},
}

plt.figure(figsize=(6, 4))

# Use a window size that makes sense for your data density
window_length = 51  # Adjust based on your data density
poly_order = 3  # Polynomial order for smoothing

# First pass: identify global reasonable limits for all methods
all_means = []
for method, props in methods.items():
    column_name = f"{method} - Average Return"
    mean = df[column_name].copy()
    all_means.extend(mean.values)

global_mean = np.nanmean(all_means)
global_std = np.nanstd(all_means)
global_upper_limit = global_mean + 3 * global_std
global_lower_limit = global_mean - 3 * global_std

for method, props in methods.items():
    column_name = f"{method} - Average Return"
    mean = df[column_name].copy()
    max_val = df[f"{column_name}__MAX"].copy()
    min_val = df[f"{column_name}__MIN"].copy()

    # Use global limits for outlier detection to ensure consistency
    mean[mean > global_upper_limit] = np.nan
    mean[mean < global_lower_limit] = np.nan
    mean = mean.interpolate(method="linear")

    max_val[max_val > global_upper_limit + global_std] = np.nan
    max_val = max_val.interpolate(method="linear")

    min_val[min_val < global_lower_limit - global_std] = np.nan
    min_val = min_val.interpolate(method="linear")

    # Apply Savitzky-Golay filter for smoothing
    try:
        smoothed_mean = savgol_filter(mean, window_length, poly_order)
        smoothed_max = savgol_filter(max_val, window_length, poly_order)
        smoothed_min = savgol_filter(min_val, window_length, poly_order)
    except:
        # If the window length is too large for the data
        smoothed_mean = mean.rolling(window=min(15, len(mean) // 5), center=True).mean()
        smoothed_max = max_val.rolling(
            window=min(15, len(max_val) // 5), center=True
        ).mean()
        smoothed_min = min_val.rolling(
            window=min(15, len(min_val) // 5), center=True
        ).mean()

    std_dev = (smoothed_max - smoothed_min) / 4

    plt.plot(
        df["Step"],
        smoothed_mean,
        label=props["label"],
        color=props["color"],
        linewidth=2,
    )
    plt.fill_between(
        df["Step"],
        smoothed_mean - std_dev,
        smoothed_mean + std_dev,
        color=props["color"],
        alpha=0.15,
    )

max_step = round(df["Step"].max())

plt.xlabel("Step (1e7)", fontsize=12)
plt.legend(fontsize=9)

plt.xticks(range(0, max_step + 1, 1))
plt.ylim(bottom=25, top=39)

# Ensure the x-axis starts at 0
plt.xlim(left=0)

plt.tight_layout()

plt.savefig("smoothed_convergence.png", dpi=300, bbox_inches="tight")
plt.close()
