import itertools

from arguments import parse_num_instances, get_arguments
from commutative_rl import BasicDQN, CommutativeDQN, HallucinatedDQN

if __name__ == '__main__':
    num_instances, remaining_argv = parse_num_instances()
    seed, approaches, max_elements, action_dims, problem_instances, reward_type, reward_noise = get_arguments(num_instances, remaining_argv)

    approach_map = {
        'BasicDQN': BasicDQN,
        'CommutativeDQN': CommutativeDQN,
        'HallucinatedDQN': HallucinatedDQN
    }

    approaches = [approach_map[name](seed, num_instances, max_elements, action_dims, reward_type, reward_noise) for name in approaches]
    learned_set = {approach.name: {problem_instance: None for problem_instance in problem_instances} for approach in approaches}

    for approach, problem_instance in itertools.product(approaches, problem_instances):
        learned_set[approach.name][problem_instance] = approach.generate_target_set(problem_instance)