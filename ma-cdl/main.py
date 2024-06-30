import itertools

from arguments import get_arguments
from crl import BasicDQN, CommutativeDQN, HallucinatedDQN


if __name__ == '__main__':    
    seed, names, num_sets, max_action, action_dims, problem_instances = get_arguments()

    basic_dqn = BasicDQN(seed, num_sets, max_action, action_dims)
    commutative_dqn = CommutativeDQN(seed, num_sets, max_action, action_dims)
    hallucinated_dqn = HallucinatedDQN(seed, num_sets, max_action, action_dims)
    
    approaches = [basic_dqn, commutative_dqn, hallucinated_dqn]
    learned_set = {approach.name: {problem_instance: None for problem_instance in problem_instances} for approach in approaches}

    for approach, problem_instance in itertools.product(approaches, problem_instances):
        learned_set[approach.name][problem_instance] = approach.generate_target_set(problem_instance)
