import yaml
import numpy as np
        

def generate_random_problems(rng: np.random.Generator, max_sum: int, action_dims: int, num_instances: int, filename: str) -> None:
    summed_values = []
    beta_a, beta_b = 3, 2

    for _ in range(num_instances):
        skewed_ratio = rng.beta(beta_a, beta_b)
        
        summed_value = int(skewed_ratio * max_sum)
        summed_values.append(summed_value)

    data = {
        'parameters': {
            'max_sum': max_sum,
            'action_dims': action_dims,
            'num_instances': num_instances
        },
        'instances': {f"instance_{i}": summed_val for i, summed_val in enumerate(summed_values)}
    }
    
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)