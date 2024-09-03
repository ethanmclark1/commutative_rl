import yaml
import numpy as np


def generate_random_problems(
    rng: np.random.Generator,
    min_dist_bounds: int,
    action_dims: int,
    negative_actions: bool,
    duplicate_actions: bool,
    num_instances: int,
    filename: str,
) -> None:
    lb_bounds, ub_bounds = 75, 200

    if negative_actions:
        lb_actions, ub_actions = -15, 5 + action_dims
    else:
        lb_actions, ub_actions = 0, 20 + action_dims

    actions_range = np.arange(lb_actions, ub_actions)

    problems = []
    for _ in range(num_instances):
        while True:
            bounds = rng.integers(lb_bounds, ub_bounds, 2)
            ub_target = int(max(bounds))
            lb_target = int(min(bounds))
            if ub_target - lb_target >= min_dist_bounds:
                break

        target_sum = int(np.median(np.arange(lb_target, ub_target)))
        actions = [
            int(action)
            for action in rng.choice(
                actions_range, action_dims, replace=duplicate_actions
            )
        ]

        if 0 not in actions:
            actions[0] = 0

        problem = {
            "sum": target_sum,
            "ub": ub_target,
            "lb": lb_target,
            "actions": actions,
        }
        problems.append(problem)

    data = {
        "parameters": {
            "min_dist_bounds": min_dist_bounds,
            "action_dims": action_dims,
            "num_instances": num_instances,
            "negative_actions": bool(negative_actions),
            "duplicate_actions": bool(duplicate_actions),
        },
        "instances": {f"instance_{i}": problem for i, problem in enumerate(problems)},
    }

    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)
