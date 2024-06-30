import yaml
import random

def generate_random_targets(max_elements: int, action_dims: int, num_instances: int, filename: str) -> list:
    groups = []
    action_set = list(range(1, action_dims))

    for _ in range(num_instances):
        non_zero_count = random.randint(1, max_elements - 1)        
        non_zero_elements = [random.choice(action_set) for _ in range(non_zero_count)]        
        non_zero_elements.sort()        
        group = non_zero_elements + [0] * (max_elements - non_zero_count)
        groups.append(group)

    non_zero_counts = set(max_elements - group.count(0) for group in groups)
    while len(non_zero_counts) < num_instances:
        index_to_change = random.randint(0, num_instances - 1)
        new_non_zero_count = random.randint(1, max_elements - 1)
        while new_non_zero_count in non_zero_counts:
            new_non_zero_count = random.randint(1, max_elements - 1)
        new_group = random.sample(action_set, new_non_zero_count)
        new_group.sort()
        new_group += [0] * (max_elements - new_non_zero_count)
        groups[index_to_change] = new_group
        non_zero_counts = set(max_elements - group.count(0) for group in groups)
        
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