import yaml
import numpy as np
import networkx as nx

from itertools import product


def generate_starts_goals(
    grid_dims: tuple, n_starts: int, n_goals: int, rng: np.random.default_rng
) -> tuple:
    width, height = grid_dims

    start_x_positions = rng.choice(range(3), size=n_starts, replace=False)
    start_y_positions = rng.choice(range(height), size=n_starts, replace=False)
    starts = [(x, y) for x, y in zip(start_x_positions, start_y_positions)]

    goal_x_positions = rng.choice(
        range(3 * width // 4, width), size=n_goals, replace=False
    )
    goal_y_positions = rng.choice(range(height), size=n_goals, replace=False)
    goals = [(x, y) for x, y in zip(goal_x_positions, goal_y_positions)]

    return starts, goals


def generate_holes(
    grid_dims: tuple, starts: list, goals: list, rng: np.random.default_rng
) -> tuple:

    width, height = grid_dims
    protected_cells = starts + goals
    n_path_pairs = len(starts) * len(goals)
    path_exists = n_path_pairs * [False]

    while True:
        holes = []

        vertical_barriers = [3]
        while vertical_barriers[-1] + 5 < width:
            vertical_barriers.append(
                vertical_barriers[-1] + rng.integers(2, 4)
            )  # 2-3 cells apart

        for vertical_barrier, y in product(vertical_barriers, range(height)):
            if (vertical_barrier, y) not in protected_cells:
                holes.append((vertical_barrier, y))

        for vertical_barrier in vertical_barriers:
            n_openings = rng.integers(1, 2, endpoint=True)
            openings = rng.choice(range(height), size=n_openings, replace=False)
            for y in openings:
                holes.remove((vertical_barrier, y))

        graph = nx.grid_graph(dim=grid_dims)
        for hole in holes:
            if hole in graph:
                graph.remove_node(hole)

        for idx, (start, goal) in enumerate(product(starts, goals)):
            path_exists[idx] = nx.has_path(graph, start, goal)

        if all(path_exists):
            break

    return holes


def generate_bridges(
    grid_dims: tuple,
    rng: np.random.default_rng,
    starts: list,
    goals: list,
    holes: list,
    n_bridges: int,
) -> list:

    base_graph = nx.grid_graph(dim=grid_dims)

    for hole in holes:
        if hole in base_graph:
            base_graph.remove_node(hole)

    baseline_lengths = {}
    for start, goal in product(starts, goals):
        path = nx.shortest_path(base_graph, start, goal)
        baseline_lengths[(start, goal)] = len(path)

    potential_bridges = []
    bridge_locations = []
    for hole in holes:
        bridge_graph = base_graph.copy()
        bridge_graph.add_node(hole)
        # Connect bridge to neighboring cells
        x, y = hole
        for neighbor_x, neighbor_y in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if (neighbor_x, neighbor_y) in bridge_graph:
                bridge_graph.add_edge(hole, (neighbor_x, neighbor_y))

        improvement = 0
        for start, goal in product(starts, goals):
            new_path_len = len(nx.shortest_path(bridge_graph, start, goal))
            baseline_len = baseline_lengths[(start, goal)]

            if new_path_len < baseline_len:
                improvement += baseline_len - new_path_len

        potential_bridges.append((hole, improvement))

    assert potential_bridges, "No potential bridges found"

    potential_bridges.sort(key=lambda x: x[1], reverse=True)

    n_high_bridges = rng.integers(3, 7)
    high_value_count = 30
    high_bridge_idxs = rng.integers(high_value_count, size=n_high_bridges)
    high_bridges = [potential_bridges[i][0] for i in high_bridge_idxs]
    bridge_values = [potential_bridges[i][1] for i in high_bridge_idxs]

    n_medium_and_high = rng.integers(10, 20)
    n_medium_bridges = n_medium_and_high - n_high_bridges

    medium_start = high_value_count
    for idx, potential_bridge in enumerate(potential_bridges):
        if potential_bridge[1] < 5:
            medium_end = idx
            break

    medium_bridge_idxs = rng.integers(medium_start, medium_end, size=n_medium_bridges)
    medium_bridges = [potential_bridges[i][0] for i in medium_bridge_idxs]
    bridge_values += [potential_bridges[i][1] for i in medium_bridge_idxs]

    # Take remaining as low-value bridges
    low_start = medium_end
    low_count = n_bridges - n_medium_and_high
    low_bridges_idx = rng.integers(low_start, len(potential_bridges), size=low_count)
    low_bridges = [potential_bridges[i][0] for i in low_bridges_idx]
    bridge_values += [potential_bridges[i][1] for i in low_bridges_idx]

    bridge_locations = high_bridges + medium_bridges + low_bridges

    assert len(bridge_locations) == n_bridges, "Incorrect number of bridges"

    return bridge_locations


def generate_problem(
    grid_dims: tuple,
    rng: np.random.default_rng,
    n_starts: int,
    n_goals: int,
    n_bridges: int,
) -> tuple:

    starts, goals = generate_starts_goals(grid_dims, n_starts, n_goals, rng)
    holes = generate_holes(grid_dims, starts, goals, rng)
    bridge_locations = generate_bridges(grid_dims, rng, starts, goals, holes, n_bridges)

    return starts, goals, holes, bridge_locations


def generate_random_problems(
    n_instances: int,
    grid_dims: tuple,
    rng: np.random.Generator,
    n_starts: int,
    n_goals: int,
    n_bridges: int,
    filename: str,
) -> None:

    problems = []
    for _ in range(n_instances):
        (starts, goals, holes, bridge_locations) = generate_problem(
            grid_dims, rng, n_starts, n_goals, n_bridges
        )

        starts = [[int(s[0]), int(s[1])] for s in starts]
        goals = [[int(g[0]), int(g[1])] for g in goals]
        holes = [[int(h[0]), int(h[1])] for h in holes]
        bridge_locations = [[int(b[0]), int(b[1])] for b in bridge_locations]

        problem = {
            "starts": starts,
            "goals": goals,
            "holes": holes,
            "bridge_locations": bridge_locations,
        }
        problems.append(problem)

    data = {
        "parameters": {
            "grid_dims": list(grid_dims),
            "n_instances": n_instances,
            "n_starts": n_starts,
            "n_goals": n_goals,
            "n_bridges": n_bridges,
        },
        "instances": {f"instance_{i}": problem for i, problem in enumerate(problems)},
    }

    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)
