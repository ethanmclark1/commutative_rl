import itertools

from arguments import parse_num_instances, get_arguments
from agents.commutative_rl import BasicDQN, CommutativeDQN, HallucinatedDQN
    

if __name__ == '__main__':
    num_instances, remaining_argv = parse_num_instances()
    seed, approaches, problem_instances, reward_type, noise_type = get_arguments(num_instances, remaining_argv)
    
    instance_nums = [int(problem_instance.split('_')[-1]) for problem_instance in problem_instances]
    small_map = any(num < 30 for num in instance_nums)
    big_map = any(num >= 30 for num in instance_nums)
    invalid_nums = small_map and big_map
    if invalid_nums:
        raise ValueError("Must have all instance nums between [0,29] or all between [30, 39] for small and big problem size respectively")
    
    map_size = '4x4' if small_map else '8x8'
    
    approach_map = {
        'BasicDQN': BasicDQN,
        'CommutativeDQN': CommutativeDQN,
        'HallucinatedDQN': HallucinatedDQN
    }
    
    approaches = [approach_map[name](seed, map_size, num_instances, reward_type, noise_type) for name in approaches]
    learned_set = {approach.name: {problem_instance: None for problem_instance in problem_instances} for approach in approaches}
    
    for approach, problem_instance in itertools.product(approaches, problem_instances):
        learned_set[approach.name][problem_instance] = approach.get_adaptations(problem_instance)