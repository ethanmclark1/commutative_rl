import yaml
import random

def generate_random_targets(max_action: int, action_dims: int, num_sets: int, filename: str) -> list:
    groups = []
    action_set = list(range(1, action_dims))

    for _ in range(num_sets):
        non_zero_count = random.randint(1, max_action - 1)        
        non_zero_elements = [random.choice(action_set) for _ in range(non_zero_count)]        
        non_zero_elements.sort()        
        group = non_zero_elements + [0] * (max_action - non_zero_count)
        groups.append(group)

    non_zero_counts = set(max_action - group.count(0) for group in groups)
    while len(non_zero_counts) < num_sets:
        index_to_change = random.randint(0, num_sets - 1)
        new_non_zero_count = random.randint(1, max_action - 1)
        while new_non_zero_count in non_zero_counts:
            new_non_zero_count = random.randint(1, max_action - 1)
        new_group = random.sample(action_set, new_non_zero_count)
        new_group.sort()
        new_group += [0] * (max_action - new_non_zero_count)
        groups[index_to_change] = new_group
        non_zero_counts = set(max_action - group.count(0) for group in groups)
        
    data = {
        'parameters': {
            'max_action': max_action,
            'action_dims': action_dims,
            'num_sets': num_sets
        },
        'instances': {f"instance_{i}": group for i, group in enumerate(groups)}
    }
    
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)