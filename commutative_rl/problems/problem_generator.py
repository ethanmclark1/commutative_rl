import yaml
import numpy as np


def generate_random_problems(rng: np.random.Generator, max_elements: int, action_dims: int, num_instances: int, filename: str) -> None:
    groups = []
    action_set = list(range(1, action_dims))

    for _ in range(num_instances):
        non_zero_count = int(rng.integers(1, max_elements)) 
        non_zero_elements = rng.choice(action_set, non_zero_count)
        non_zero_elements.sort()
        group = [int(x) for x in non_zero_elements] + [0] * (max_elements - non_zero_count)
        groups.append(group)

    data = {
        'parameters': {
            'max_elements': max_elements,
            'action_dims': action_dims,
            'num_instances': num_instances
        },
        'instances': {f"instance_{i}": group for i, group in enumerate(groups)}
    }
    
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)