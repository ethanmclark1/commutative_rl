import yaml
import numpy as np


def generate_random_problems(
    rng: np.random.Generator,
    sum_range: int,
    elems_range: int,
    n_elems: int,
    num_instances: int,
    filename: str,
) -> None:

    problems = []
    for _ in range(num_instances):
        sum = int(rng.choice(sum_range))
        elements = rng.choice(elems_range, size=n_elems, replace=False)
        elements = [int(e) for e in elements]
        elements[0] = 0

        elements_cost = rng.choice(elems_range, size=n_elems)
        elements_cost = [int(e) / sum for e in elements_cost]
        elements_cost[0] = 0

        problem = {
            "sum": sum,
            "elements": elements,
            "elements_cost": elements_cost,
        }
        problems.append(problem)

    data = {
        "parameters": {
            "sum_range": [sum_range[0], sum_range[-1]],
            "elems_range": [elems_range[0], elems_range[-1]],
            "n_elems": n_elems,
            "num_instances": num_instances,
        },
        "instances": {f"instance_{i}": problem for i, problem in enumerate(problems)},
    }

    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)
