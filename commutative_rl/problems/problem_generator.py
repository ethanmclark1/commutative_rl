import yaml
import numpy as np
import networkx as nx

from itertools import product
from agents.utils.helpers import visualize_grid


def generate_starts_goals(
    grid_dims: tuple, n_starts: int, n_goals: int, rng: np.random.default_rng
) -> tuple:

    starts = []
    goals = []

    width, height = grid_dims

    while len(starts) < n_starts:
        # Start positions are in leftmost part of grid
        start = (rng.integers(0, 2), rng.integers(0, height))
        if start not in (starts):
            starts.append(start)

    while len(goals) < n_goals:
        # Goal positions are in rightmost part of grid
        goal = (width - 1, rng.integers(0, height))
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
    _, height = grid_dims

    # Create vertical barriers with gaps that alternate between top and bottom
    for i, x in enumerate(barrier_positions):
        gap_positions = height - 1 if i % 2 == 0 else 0
        for y in range(height):
            if y != gap_positions and (x, y) not in (starts + goals):
                holes.add((x, y))

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

    _, height = grid_dims

    graph_with_holes = nx.grid_2d_graph(*grid_dims)
    for hole in holes:
        if hole in graph_with_holes:
            graph_with_holes.remove_node(hole)

    baseline_paths = {}
    path_pairs = list(product(starts, goals))

    for start, goal in path_pairs:
        path_len = nx.shortest_path_length(graph_with_holes, start, goal)
        baseline_paths[(start, goal)] = path_len

    independent_bridges = []
    n_independent = n_bridges // 3  # 1/3 of bridges have standalone value

    barrier_indices = list(range(len(barrier_positions)))
    rng.shuffle(barrier_indices)

    for barrier_idx in barrier_indices:
        if len(independent_bridges) >= n_independent:
            break

        barrier_x = barrier_positions[barrier_idx]

        height = grid_dims[1]
        y_positions = list(range(1, height - 1))  # Avoid edges
        rng.shuffle(y_positions)

        for y in y_positions:
            bridge_pos = (barrier_x, y)

            if bridge_pos not in holes:
                continue

            test_graph = graph_with_holes.copy()
            test_graph.add_node(bridge_pos)

            neighbors = [
                (bridge_pos[0] + 1, bridge_pos[1]),
                (bridge_pos[0] - 1, bridge_pos[1]),
                (bridge_pos[0], bridge_pos[1] + 1),
                (bridge_pos[0], bridge_pos[1] - 1),
            ]
            for neighbor in neighbors:
                if neighbor in test_graph:
                    test_graph.add_edge(bridge_pos, neighbor)

            # Check if this bridge helps at least one path significantly
            has_independent_value = False
            for start, goal in path_pairs:
                original_len = baseline_paths.get((start, goal))
                new_len = nx.shortest_path_length(test_graph, start, goal)
                if original_len - new_len > 2:
                    has_independent_value = True
                    break

            if has_independent_value:
                independent_bridges.append(bridge_pos)
                break

    current_graph = graph_with_holes.copy()
    selected_bridges = independent_bridges.copy()
    # Add in all independent bridges to current graph
    for bridge in selected_bridges:
        current_graph.add_node(bridge)
        neighbors = [
            (bridge[0] + 1, bridge[1]),
            (bridge[0] - 1, bridge[1]),
            (bridge[0], bridge[1] + 1),
            (bridge[0], bridge[1] - 1),
        ]
        for neighbor in neighbors:
            if neighbor in current_graph:
                current_graph.add_edge(bridge, neighbor)

    # Calculate current path lengths with independent bridges
    current_paths = {}
    for start, goal in path_pairs:
        path = nx.shortest_path_length(current_graph, start, goal)
        current_paths[(start, goal)] = path

    remaining_candidates = [
        h for h in holes if h not in selected_bridges and h[1] not in [0, height - 1]
    ]
    rng.shuffle(remaining_candidates)

    while len(selected_bridges) < n_bridges and remaining_candidates:
        best_bridge = None
        best_value = -1

        for candidate in remaining_candidates:
            # Create test graph with this additional bridge
            test_graph = current_graph.copy()
            test_graph.add_node(candidate)

            neighbors = [
                (candidate[0] + 1, candidate[1]),
                (candidate[0] - 1, candidate[1]),
                (candidate[0], candidate[1] + 1),
                (candidate[0], candidate[1] - 1),
            ]
            for neighbor in neighbors:
                if neighbor in test_graph:
                    test_graph.add_edge(candidate, neighbor)

            # Calculate how much this bridge improves paths in combination with independent bridges
            synergy_value = 0
            for start, goal in path_pairs:
                current_len = current_paths.get((start, goal))
                new_len = nx.shortest_path_length(test_graph, start, goal)
                improvement = current_len - new_len
                synergy_value += improvement

            if synergy_value > best_value:
                best_value = synergy_value
                best_bridge = candidate

        if best_bridge is None:
            break

        selected_bridges.append(best_bridge)
        remaining_candidates.remove(best_bridge)

        # Update current graph
        current_graph.add_node(best_bridge)
        neighbors = [
            (best_bridge[0] + 1, best_bridge[1]),
            (best_bridge[0] - 1, best_bridge[1]),
            (best_bridge[0], best_bridge[1] + 1),
            (best_bridge[0], best_bridge[1] - 1),
        ]
        for neighbor in neighbors:
            if neighbor in current_graph:
                current_graph.add_edge(best_bridge, neighbor)

        for start, goal in path_pairs:
            path_len = nx.shortest_path_length(current_graph, start, goal)
            current_paths[(start, goal)] = path_len

    return selected_bridges


def generate_problem(
    grid_dims: tuple,
    n_starts: int,
    n_goals: int,
    n_bridges: int,
    rng: np.random.default_rng,
) -> tuple:

    width, _ = grid_dims
    graph = nx.grid_2d_graph(*grid_dims)

    n_barriers = 5
    min_barrier_width = 1
    barrier_positions = np.linspace(width // 6, 5 * width // 6, n_barriers, dtype=int)

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

    bridge_locations = generate_bridges(
        grid_dims, starts, goals, holes, n_bridges, barrier_positions, rng
    )

    visualize_grid(grid_dims, starts, goals, holes, bridge_locations)

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
    while len(problems) < n_instances:
        starts, goals, holes, bridge_locations = generate_problem(
            grid_dims, n_starts, n_goals, n_bridges, rng
        )

        # Prepare data for YAML serialization
        starts = [[int(start[0]), int(start[1])] for start in starts]
        goals = [[int(goal[0]), int(goal[1])] for goal in goals]
        holes = [[int(hole[0]), int(hole[1])] for hole in holes]
        bridge_locations = [[int(loc[0]), int(loc[1])] for loc in bridge_locations]

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
            "n_starts": n_starts,
            "n_goals": n_goals,
            "n_bridges": n_bridges,
            "n_instances": n_instances,
        },
        "instances": {f"instance_{i}": problem for i, problem in enumerate(problems)},
    }

    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)
