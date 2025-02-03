import yaml
import numpy as np
import networkx as nx


desc = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]


def random_num_in_range(rng, low, high):
    random_val = rng.random()
    val_in_range = random_val * (high - low) + low
    return val_in_range


def generate_random_problems(
    rng: np.random.Generator,
    grid_dims: tuple,
    n_starts: int,
    n_goals: int,
    n_bridges: int,
    n_holes: int,
    n_instances: int,
    filename: str,
) -> None:

    problems = []
    while len(problems) < n_instances:
        start, goal, holes, bridge_locations, bridge_costs = generate_problem(
            grid_dims, n_starts, n_goals, n_bridges, n_holes, rng
        )

        G = nx.grid_2d_graph(*grid_dims)
        for hole in holes:
            G.remove_node(tuple(hole))

        if all(nx.has_path(G, tuple(s), tuple(g)) for s in start for g in goal):
            problem = {
                "starts": start,
                "goals": goal,
                "holes": holes,
                "bridge_locations": bridge_locations,
                "bridge_costs": bridge_costs,
            }
            problems.append(problem)

    data = {
        "parameters": {
            "grid_dims": list(grid_dims),
            "n_starts": n_starts,
            "n_goals": n_goals,
            "n_bridges": n_bridges,
            "n_holes": n_holes,
            "n_instances": n_instances,
        },
        "instances": {f"instance_{i}": problem for i, problem in enumerate(problems)},
    }

    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def generate_problem(
    grid_size: tuple,
    n_starts: int,
    n_goals: int,
    n_bridges: int,
    n_holes: int,
    rng: np.random.Generator,
) -> tuple:
    # Checks if source is not in the list of args
    def free_spot(source, *args):
        return source not in args

    max_iterations = 25
    width, height = grid_size
    graph = nx.grid_2d_graph(width, height)

    starts, goals, holes = [], [], []

    while len(starts) < n_starts:
        start = rng.integers(0, width), rng.integers(0, height)
        if free_spot(start, *starts):
            starts.append((int(start[0]), int(start[1])))

    while len(goals) < n_goals:
        goal = rng.integers(0, width), rng.integers(0, height)
        if free_spot(goal, *starts, *goals):
            goals.append((int(goal[0]), int(goal[1])))

    iteration_count = 0
    while len(holes) < n_holes:
        iteration_count += 1
        if iteration_count > max_iterations:  # Stuck in an infinite loop
            for _ in range(n_holes - len(holes)):
                while True:
                    hole = tuple(rng.choice(list(graph.nodes)))
                    if free_spot(hole, *starts, *goals, *holes):
                        tmp_graph = graph.copy()
                        tmp_graph.remove_node(hole)
                        if all(
                            nx.has_path(tmp_graph, s, g) for s, g in zip(starts, goals)
                        ):
                            holes.append((int(hole[0]), int(hole[1])))
                            graph.remove_node(hole)
                            break
            break
        # Generate a hole in a location that blocks the shortest path
        for start, goal in zip(starts, goals):
            path = nx.shortest_path(graph, start, goal)
            available_pos = [
                x for x in path if x not in starts and x not in goals and x not in holes
            ]
            if available_pos:
                hole = tuple(rng.choice(available_pos))
                if free_spot(hole, *starts, *goals, *holes):
                    tmp_graph = graph.copy()
                    tmp_graph.remove_node(hole)
                    if all(nx.has_path(tmp_graph, s, g) for s, g in zip(starts, goals)):
                        holes.append((int(hole[0]), int(hole[1])))
                        graph.remove_node(hole)
                        iteration_count = 0
                        break

    bridge_locations = rng.choice(holes, n_bridges, replace=False)

    starts = [list(start) for start in starts]
    goals = [list(goal) for goal in goals]
    holes = [list(hole) for hole in holes]
    bridge_locations = [
        [int(bridge_location[0]), int(bridge_location[1])]
        for bridge_location in bridge_locations
    ]
    bridge_locations = sorted(bridge_locations)
    bridge_locations.append(0)  # add terminating action
    bridge_costs = [random_num_in_range(rng, 0, 3) for _ in range(n_bridges)]
    bridge_costs.append(0)  # add terminating action cost

    return starts, goals, holes, bridge_locations, bridge_costs
