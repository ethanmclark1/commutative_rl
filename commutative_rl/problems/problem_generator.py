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


def generate_random_problems(
    rng: np.random.Generator,
    grid_dims: tuple,
    n_bridges: int,
    num_instances: int,
    filename: str,
) -> None:

    problems = []
    for _ in range(num_instances):
        start, goal, holes, hole_probs, mapping = generate_problem(
            grid_dims, n_bridges, rng
        )

        G = nx.grid_2d_graph(*grid_dims)
        for hole in holes:
            G.remove_node(tuple(hole))

        if all(nx.has_path(G, tuple(s), tuple(g)) for s in start for g in goal):
            problem = {
                "starts": start,
                "goals": goal,
                "holes": holes,
                "hole_probs": hole_probs,
                "mapping": mapping,
            }
            problems.append(problem)

        data = {
            "parameters": {
                "grid_dims": list(grid_dims),
                "n_bridges": n_bridges,
                "num_instances": num_instances,
            },
            "instances": {
                f"instance_{i}": problem for i, problem in enumerate(problems)
            },
        }

    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def generate_problem(
    grid_size: tuple, num_bridges: int, rng: np.random.default_rng
) -> tuple:
    # Checks if source is not in the list of args
    def free_spot(source, *args):
        return source not in args

    num_starts = 4
    num_goals = rng.integers(8, 10)
    num_holes = 25

    max_iterations = 25
    width, height = grid_size
    graph = nx.grid_2d_graph(width, height)

    starts, goals, holes, hole_probs = [], [], [], []

    while len(starts) < num_starts:
        start = rng.integers(0, width), rng.integers(0, height)
        if free_spot(start, *starts):
            starts.append((int(start[0]), int(start[1])))

    while len(goals) < num_goals:
        goal = rng.integers(0, width), rng.integers(0, height)
        if free_spot(goal, *starts, *goals):
            goals.append((int(goal[0]), int(goal[1])))

    iteration_count = 0
    while len(holes) < num_holes:
        iteration_count += 1
        if iteration_count > max_iterations:  # Stuck in an infinite loop
            for _ in range(num_holes - len(holes)):
                while True:
                    hole = tuple(rng.choice(list(graph.nodes)))
                    if free_spot(hole, *starts, *goals, *holes):
                        tmp_graph = graph.copy()
                        tmp_graph.remove_node(hole)
                        if all(
                            nx.has_path(tmp_graph, s, g) for s, g in zip(starts, goals)
                        ):
                            holes.append((int(hole[0]), int(hole[1])))
                            hole_probs.append(
                                float(np.clip(rng.normal(loc=0.75, scale=0.15), 0, 1))
                            )
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
                        hole_probs.append(
                            float(np.clip(rng.normal(loc=0.7, scale=0.2), 0, 1))
                        )
                        graph.remove_node(hole)
                        iteration_count = 0
                        break

    bridges = rng.choice(holes, num_bridges, replace=False)

    starts = [list(start) for start in starts]
    goals = [list(goal) for goal in goals]
    holes = [list(hole) for hole in holes]
    bridges = [[int(bridge[0]), int(bridge[1])] for bridge in bridges]
    bridges = sorted(bridges)

    mapping = {i: bridge for i, bridge in enumerate(bridges, start=1)}

    return starts, goals, holes, hole_probs, mapping
