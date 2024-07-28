import yaml
import numpy as np
import networkx as nx


desc = {
    '5x5':
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    '8x8':
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
}


def generate_instances(
    problem_size: str,
    rng: np.random.default_rng,
    grid_dims: tuple, 
    total_num_instances: int,
    action_size: int,
    output_file: str='commutative_rl/problems/problems.yaml'
    ) -> None:
    
    multiplier = 0.25 if problem_size == '8x8' else 0.75
    num_instances = int(total_num_instances * multiplier)
    
    with open(output_file, 'r') as file:
        problem_instances = yaml.safe_load(file) or {}
    
    data = problem_instances.get(problem_size, {})
    if data and len(data.keys()) == num_instances:
        return
    elif data and len(data.keys()) > num_instances:
        for i in range(num_instances, len(data.keys())):
            data.pop(f'instance_{i}')
    else:
        problems = {}
        start_shift = 0 if problem_size == '5x5' else 30
        
        if data:
            i = len(data.keys())
            problems.update(problem_instances[problem_size])
        else:
            i = 0
        
        while i < num_instances:
            start, goal, holes, hole_probs, mapping = generate_problem(grid_dims, action_size, rng)
            
            G = nx.grid_2d_graph(*grid_dims)
            for hole in holes:
                G.remove_node(tuple(hole))
            
            if all(nx.has_path(G, tuple(s), tuple(g)) for s in start for g in goal):
                problem = {
                    'starts': start,
                    'goals': goal,
                    'holes': holes,
                    'hole_probs': hole_probs,
                    'mapping': mapping
                }
                problems[f'instance_{start_shift + i}'] = problem
                i += 1
        
        problem_instances[problem_size] = problems
        
        with open(output_file, 'w') as file:
            yaml.dump(problem_instances, file)

def generate_problem(grid_size: tuple, num_bridges: int, rng: np.random.default_rng) -> tuple:
    # Checks if source is not in the list of args
    def free_spot(source, *args):
        return source not in args
    
    if grid_size == (5, 5):
        num_starts = 2
        num_goals = rng.integers(2, 4)
        num_holes = 8
    else:
        num_starts = 4
        num_goals = rng.integers(8, 10)
        num_holes =  25
    
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
        if iteration_count > max_iterations: # Stuck in an infinite loop
            for _ in range(num_holes - len(holes)):
                while True:
                    hole = tuple(rng.choice(list(graph.nodes)))
                    if free_spot(hole, *starts, *goals, *holes):
                        tmp_graph = graph.copy()
                        tmp_graph.remove_node(hole)
                        if all(nx.has_path(tmp_graph, s, g) for s, g in zip(starts, goals)):
                            holes.append((int(hole[0]), int(hole[1])))
                            hole_probs.append(float(np.clip(rng.normal(loc=0.75, scale=0.15), 0, 1)))
                            graph.remove_node(hole)
                            break
            break
        # Generate a hole in a location that blocks the shortest path
        for start, goal in zip(starts, goals):
            path = nx.shortest_path(graph, start, goal)
            available_pos = [x for x in path if x not in starts and x not in goals and x not in holes]
            if available_pos:
                hole = tuple(rng.choice(available_pos))
                if free_spot(hole, *starts, *goals, *holes):
                    tmp_graph = graph.copy()
                    tmp_graph.remove_node(hole)
                    if all(nx.has_path(tmp_graph, s, g) for s, g in zip(starts, goals)):
                        holes.append((int(hole[0]), int(hole[1])))
                        hole_probs.append(float(np.clip(rng.normal(loc=0.7, scale=0.2), 0, 1)))
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

def get_instance(problem_instance: str, output_file: str='commutative_rl/problems/problems.yaml') -> dict:
    with open(output_file, 'r') as file:
        problems = yaml.safe_load(file)
    
    instance_num = int(problem_instance.split('_')[-1])
    if instance_num < 30:
        instance = problems['5x5'][problem_instance]
    else:
        instance = problems['8x8'][problem_instance]
    
    return instance

def get_entity_positions(instance: dict, rng: np.random.default_rng, percent_holes: float) -> tuple:
    start = tuple(rng.choice(instance['starts']))
    goal = tuple(rng.choice(instance['goals']))
    
    holes = instance['holes']
    
    if percent_holes != 1:
        num_holes = int(len(holes) * percent_holes)        
        normalized_probs = np.array(instance['hole_probs']) / np.sum(instance['hole_probs'])
        holes = rng.choice(holes, num_holes, replace=False, p=normalized_probs)
        
    holes = [tuple(hole) for hole in holes]
    
    return start, goal, holes