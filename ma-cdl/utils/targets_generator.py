import yaml
import random

def generate_random_targets(max_elements: int, action_dims: int, num_instances: int, filename: str) -> None:
    action_set = list(range(1, action_dims))
    groups = []

    for _ in range(num_instances):
        # Ensure at least one zero by reducing the max non-zero count
        max_non_zero = max_elements - 1
        non_zero_count = random.randint(1, max_non_zero)
        non_zero_elements = random.choices(action_set, k=non_zero_count)
        non_zero_elements.sort()
        group = non_zero_elements + [0] * (max_elements - non_zero_count)
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