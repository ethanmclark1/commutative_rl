import itertools

from arguments import parse_num_instances, get_arguments
from agents.commutative_rl import Traditional, Commutative, Hallucinated

if __name__ == "__main__":
    num_instances, remaining_argv = parse_num_instances()
    (
        seed,
        approaches,
        max_sum,
        min_dist_bounds,
        action_dims,
        negative_actions,
        duplicate_actions,
        problem_instances,
        reward_type,
        reward_noise,
    ) = get_arguments(num_instances, remaining_argv)

    approach_map = {
        "Traditional": Traditional,
        "Commutative": Commutative,
        "Hallucinated": Hallucinated,
    }

    approaches = [
        approach_map[name](
            seed,
            num_instances,
            max_sum,
            min_dist_bounds,
            action_dims,
            negative_actions,
            duplicate_actions,
            reward_type,
            reward_noise,
        )
        for name in approaches
    ]
    learned_set = {
        approach.name: {
            problem_instance: None for problem_instance in problem_instances
        }
        for approach in approaches
    }

    for approach, problem_instance in itertools.product(approaches, problem_instances):
        learned_set[approach.name][problem_instance] = approach.generate_target_sum(
            problem_instance
        )
