import numpy as np
import matplotlib.pyplot as plt


def visualize_grid(
    grid_dims: tuple,
    starts: list,
    goals: list,
    holes: list,
    bridge_locations: list,
    paths: list = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))

    width, height = grid_dims
    for x in range(width + 1):
        ax.axvline(x, color="gray", linewidth=0.5)
    for y in range(height + 1):
        ax.axhline(y, color="gray", linewidth=0.5)

    if paths:
        for path in paths:
            # Extract x and y coordinates, adding 0.5 to center within grid cells
            path_x = [pos[0] + 0.5 for pos in path]
            path_y = [pos[1] + 0.5 for pos in path]
            # Plot path with semi-transparent blue
            ax.plot(
                path_x, path_y, color="blue", alpha=0.3, linewidth=2, linestyle="--"
            )

    # Plot holes (black squares)
    holes_x = [pos[0] + 0.5 for pos in holes if pos != [0, 0]]
    holes_y = [pos[1] + 0.5 for pos in holes if pos != [0, 0]]
    ax.scatter(holes_x, holes_y, color="black", s=400, marker="s", label="Holes")

    # Plot bridges (orange squares)
    bridge_locations_x = [pos[0] + 0.5 for pos in bridge_locations if pos != 0]
    bridge_locations_y = [pos[1] + 0.5 for pos in bridge_locations if pos != 0]
    ax.scatter(
        bridge_locations_x,
        bridge_locations_y,
        color="orange",
        s=400,
        marker="s",
        label="Bridges",
    )

    starts_x = [pos[0] + 0.5 for pos in starts]
    starts_y = [pos[1] + 0.5 for pos in starts]
    ax.scatter(starts_x, starts_y, color="green", s=300, marker="^", label="Starts")

    goals_x = [pos[0] + 0.5 for pos in goals]
    goals_y = [pos[1] + 0.5 for pos in goals]
    ax.scatter(goals_x, goals_y, color="red", s=300, marker="D", label="Goals")

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.title("Grid Visualization")
    plt.tight_layout()
    plt.show()


def random_num_in_range(rng: np.random.default_rng, low: float, high: float) -> float:
    random_val = rng.random()
    val_in_range = random_val * (high - low) + low
    return val_in_range


def encode(state: float, n_states: int) -> float:
    binary_str = reversed([str(cell) for cell in state])
    binary_str = "".join(binary_str)
    state = int(binary_str, 2)

    state = state / n_states

    return state


def decode(state: float, n_bridges: int, n_states: int) -> np.ndarray:
    state = int(state * n_states)

    binary_str = bin(state)[2:].zfill(n_bridges)
    state = np.array([int(cell) for cell in reversed(binary_str)], dtype=np.int64)

    return state


def argmax(array: np.ndarray, action_rng: np.random.default_rng) -> int:
    try:
        return action_rng.choice(np.where(array == np.max(array))[0])
    except:
        return np.argmax(array)
