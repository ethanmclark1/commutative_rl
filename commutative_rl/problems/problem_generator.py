import yaml
import numpy as np


def random_num_in_range(rng: np.random.Generator, num_range: range) -> float:
    random_val = rng.random()
    val_in_range = random_val * (num_range.stop - num_range.start) + num_range.start
    return val_in_range


def generate_random_problems(
    rng: np.random.Generator,
    sum_range: range,
    elems_range: range,
    n_elems: int,
    num_instances: int,
    filename: str,
) -> None:

    # generate discrete problems for qtable
    qtable_problems = []
    for _ in range(num_instances):
        target_sum = int(rng.choice(sum_range))
        elements = [int(rng.choice(elems_range)) for i in range(n_elems)]
        elements[-1] = 0  # terminating action
        element_costs = [int(rng.choice(elems_range)) for i in range(n_elems - 1)]

        problem = {
            "target_sum": target_sum,
            "elements": elements,
            "element_costs": element_costs,
        }
        qtable_problems.append(problem)

    # generate continuous problems for dqn
    dqn_problems = []
    for _ in range(num_instances):
        target_sum = random_num_in_range(rng, sum_range)
        elements = [random_num_in_range(rng, elems_range) for i in range(n_elems)]
        elements[-1] = 0  # terminating action
        element_costs = [
            random_num_in_range(rng, elems_range) for i in range(n_elems - 1)
        ]

        problem = {
            "target_sum": target_sum,
            "elements": elements,
            "element_costs": element_costs,
        }
        dqn_problems.append(problem)

    data = {
        "qtable": {
            "parameters": {
                "sum_range": [sum_range.start, sum_range.stop],
                "elems_range": [elems_range.start, elems_range.stop],
                "n_elems": n_elems,
                "num_instances": num_instances,
            },
            "instances": {
                f"instance_{i}": problem for i, problem in enumerate(qtable_problems)
            },
        },
        "dqn": {
            "parameters": {
                "sum_range": [sum_range.start, sum_range.stop],
                "elems_range": [elems_range.start, elems_range.stop],
                "n_elems": n_elems,
                "num_instances": num_instances,
            },
            "instances": {
                f"instance_{i}": problem for i, problem in enumerate(dqn_problems)
            },
        },
    }

    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)
