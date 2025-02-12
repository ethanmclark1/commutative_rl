import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product
from collections import defaultdict


def visualize_grid(
    starts: list,
    goals: list,
    holes: list,
    bridge_locations: list,
    grid_dims: tuple,
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


def generate_starts_goals(
    grid_dims: tuple, n_starts: int, n_goals: int, rng: np.random.default_rng
) -> tuple:

    starts = []
    goals = []

    width, height = grid_dims

    while len(starts) < n_starts:
        # Start positions are in leftmost quarter of grid
        start = (rng.integers(0, width // 4), rng.integers(0, height))
        if start not in (starts):
            starts.append(start)

    while len(goals) < n_goals:
        # Goal positions are in rightmost quarter of grid
        goal = (rng.integers(3 * width // 4, width), rng.integers(0, height))
        if goal not in (starts + goals):
            goals.append(goal)

    return starts, goals


def generate_holes(
    G: nx.Graph,
    n_bridges: int,
    starts: list,
    goals: list,
    min_barrier_width: float,
    barrier_positions: list,
    grid_dims: tuple,
    rng: np.random.default_rng,
) -> list:

    holes = set()

    width, height = grid_dims

    # Create vertical barriers with gaps
    for x in barrier_positions:
        barrier_height = height - rng.integers(2, 6)  # Leave some space at top/bottom
        gap_positions = rng.integers(0, barrier_height, size=2)  # 2 gaps per barrier

        for y in range(barrier_height):
            if y not in gap_positions and not any(
                abs(y - gap) <= 1 for gap in gap_positions
            ):
                if (x, y) not in (starts + goals):
                    holes.add((x, y))
                    # Add random width to barrier
                    for dx in range(1, min_barrier_width):
                        if rng.random() < 0.7 and (x + dx, y) not in (starts + goals):
                            holes.add((x + dx, y))

    # Add diagonal connectors between barriers
    for i in range(len(barrier_positions) - 1):
        x1, x2 = barrier_positions[i], barrier_positions[i + 1]
        y1, y2 = rng.integers(height // 4, 3 * height // 4, size=2)

        # Create diagonal connector
        steps = max(abs(x2 - x1), abs(y2 - y1))
        for step in range(steps):
            x = int(x1 + (x2 - x1) * step / steps)
            y = int(y1 + (y2 - y1) * step / steps)
            if (x, y) not in (starts + goals):
                holes.add((x, y))

    # Add maze-like elements in spaces between barriers
    for i in range(len(barrier_positions) - 1):
        x1, x2 = barrier_positions[i], barrier_positions[i + 1]
        center_x = (x1 + x2) // 2
        center_y = height // 2

        # Create a small maze-like pattern around center point
        for dx, dy in product(range(-3, 4), repeat=2):
            if rng.random() < 0.6:
                pos = (center_x + dx, center_y + dy)
                if 0 <= pos[0] < width and 0 <= pos[1] < height:
                    if pos not in (starts + goals):
                        holes.add(pos)

    tmp_G = G.copy()
    for hole in holes:
        tmp_G.remove_node(hole)

    valid_paths_exist = all(
        nx.has_path(tmp_G, start, goal) for start, goal in product(starts, goals)
    )

    if not valid_paths_exist:
        return generate_holes(
            G,
            n_bridges,
            starts,
            goals,
            min_barrier_width,
            barrier_positions,
            grid_dims,
            rng,
        )

    holes = sorted(list(holes))

    return holes


def generate_bridges(
    grid_dims: tuple,
    starts: list,
    goals: list,
    holes: list,
    n_bridges: int,
    barrier_positions: list,
    rng: np.random.default_rng,
) -> list:

    # Measure the utility of a cluster of holes
    def cluster_utility(cluster: list, graph: nx.Graph) -> int:
        utility = 0
        temp_graph = nx.grid_2d_graph(*grid_dims)

        holes_to_remove = [hole for hole in holes if tuple(hole) not in cluster]
        temp_graph.remove_nodes_from(holes_to_remove)

        for start, goal in product(starts, goals):
            try:
                original_len = len(nx.shortest_path(graph, start, goal))
                new_len = len(nx.shortest_path(temp_graph, start, goal))
                utility += original_len - new_len
            except nx.NetworkXNoPath:
                utility += 0

        return utility

    original_paths = []
    bridge_candidates = []
    bridge_costs = []
    graph_no_holes = nx.grid_2d_graph(*grid_dims)

    width, height = grid_dims

    top_row = height // 3
    bottom_row = 2 * height // 3

    # All holes on both target rows are automatically bridge candidates
    top_horizontal_holes = [(x, top_row) for x in range(width) if (x, top_row) in holes]
    bottom_horizontal_holes = [
        (x, bottom_row) for x in range(width) if (x, bottom_row) in holes
    ]
    bridge_candidates.extend(top_horizontal_holes)
    bridge_candidates.extend(bottom_horizontal_holes)

    for start, goal in product(starts, goals):
        path = nx.shortest_path(graph_no_holes, start, goal)
        original_paths.append(path)

    heat_map = defaultdict(int)
    for path in original_paths:
        for pos in path:
            if pos not in (starts + goals):
                heat_map[pos] += 1

    graph_with_holes = nx.grid_2d_graph(*grid_dims)
    for hole in holes:
        graph_with_holes.remove_node(tuple(hole))

    # Add additional strategic bridges in other locations
    for i in range(len(barrier_positions) - 1):
        x0, x1 = barrier_positions[i], barrier_positions[i + 1]

        for x in range(x0, x1):
            for y in range(height):
                if (
                    y in [top_row, bottom_row] or (x, y) not in holes
                ):  # Skip target rows and non-holes
                    continue

                potential_patterns = [
                    # Vertical pattern (3)
                    [(x, y), (x, y + 1), (x, y + 2)],
                    # Vertical pattern (4)
                    [(x, y), (x, y + 1), (x, y + 2), (x, y + 3)],
                    # Horizontal pattern (3)
                    [(x, y), (x + 1, y), (x + 2, y)],
                    # Horizontal pattern (4)
                    [(x, y), (x + 1, y), (x + 2, y), (x + 3, y)],
                    # Square pattern (2x2
                    [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)],
                    # L-shaped patterns
                    [(x, y), (x + 1, y), (x + 1, y + 1)],
                    [(x, y), (x, y + 1), (x + 1, y + 1)],
                ]

                for pattern in potential_patterns:
                    if all(p in holes for p in pattern):
                        utility = cluster_utility(pattern, graph_with_holes)
                        if utility > 0:
                            bridge_candidates.extend(pattern)

    bridge_candidates = list(set(bridge_candidates))

    if len(bridge_candidates) < n_bridges:
        # Add additional bridges in locations with non-zero heat map values
        potential_singles = [
            (x, y)
            for x, y in holes
            if (x, y) not in bridge_candidates and heat_map.get((x, y), 0) > 0
        ]
        while len(bridge_candidates) < n_bridges and potential_singles:
            pos = potential_singles.pop(rng.integers(len(potential_singles)))
            bridge_candidates.append(pos)

    # Ensure we keep all bridges from our target rows
    horizontal_bridges = [
        (x, y) for (x, y) in bridge_candidates if y in [top_row, bottom_row]
    ]
    other_bridges = [
        (x, y) for (x, y) in bridge_candidates if y not in [top_row, bottom_row]
    ]

    n_remaining = n_bridges - len(horizontal_bridges)
    if n_remaining > 0 and len(other_bridges) > n_remaining:
        other_selected = rng.choice(other_bridges, size=n_remaining, replace=False)
        bridge_locations = horizontal_bridges + list(other_selected)
    else:
        bridge_locations = bridge_candidates

    bridge_locations = sorted([list(loc) for loc in bridge_locations])
    bridge_costs = [random_num_in_range(rng, 0.5, 3.0) for _ in range(n_bridges)]
    bridge_locations.append(0)
    bridge_costs.append(0)

    return bridge_locations, bridge_costs


def generate_problem(
    grid_dims: tuple,
    n_starts: int,
    n_goals: int,
    n_bridges: int,
    rng: np.random.default_rng,
) -> tuple:

    width, _ = grid_dims
    graph = nx.grid_2d_graph(*grid_dims)

    n_barriers = 4
    min_barrier_width = 3
    barrier_positions = np.linspace(width // 4, 3 * width // 4, n_barriers, dtype=int)

    starts, goals = generate_starts_goals(grid_dims, n_starts, n_goals, rng)

    holes = generate_holes(
        graph,
        n_bridges,
        starts,
        goals,
        min_barrier_width,
        barrier_positions,
        grid_dims,
        rng,
    )

    bridge_locations, bridge_costs = generate_bridges(
        grid_dims, starts, goals, holes, n_bridges, barrier_positions, rng
    )

    return starts, goals, holes, bridge_locations, bridge_costs


def generate_random_problems(
    rng: np.random.Generator,
    grid_dims: tuple,
    n_starts: int,
    n_goals: int,
    n_bridges: int,
    n_instances: int,
    filename: str,
) -> None:
    problems = []
    while len(problems) < n_instances:
        starts, goals, holes, bridge_locations, bridge_costs = generate_problem(
            grid_dims, n_starts, n_goals, n_bridges, rng
        )

        # Prepare data for YAML serialization
        starts = [[int(start[0]), int(start[1])] for start in starts]
        goals = [[int(goal[0]), int(goal[1])] for goal in goals]
        holes = [[int(hole[0]), int(hole[1])] for hole in holes]
        bridge_locations = [
            [int(loc[0]), int(loc[1])] if loc != 0 else 0 for loc in bridge_locations
        ]

        problem = {
            "starts": starts,
            "goals": goals,
            "holes": holes,
            "bridge_locations": bridge_locations,
            "bridge_costs": bridge_costs,
        }
        problems.append(problem)

        visualize_grid(starts, goals, holes, bridge_locations, grid_dims)

    data = {
        "parameters": {
            "grid_dims": list(grid_dims),
            "n_starts": n_starts,
            "n_goals": n_goals,
            "n_bridges": n_bridges,
            "n_instances": n_instances,
        },
        "instances": {f"instance_{i}": problem for i, problem in enumerate(problems)},
    }

    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)
