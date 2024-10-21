import itertools

from agents.traditional import Traditional, TripleTraditional
from agents.commutative import (
    Commutative,
)
from arguments import parse_num_instances, get_arguments


if __name__ == "__main__":
    num_instances, remaining_argv = parse_num_instances()
    (seed, approaches, problem_instances, noise_type) = get_arguments(
        num_instances, remaining_argv
    )

    approach_map = {
        "Traditional": Traditional,
        "Commutative": Commutative,
        "TripleTraditional": TripleTraditional,
    }

    approaches = [
        approach_map[name](seed, num_instances, noise_type) for name in approaches
    ]
    learned_set = {
        approach.name: {
            problem_instance: None for problem_instance in problem_instances
        }
        for approach in approaches
    }

    for approach, problem_instance in itertools.product(approaches, problem_instances):
        learned_set[approach.name][problem_instance] = approach.generate_city_design(
            problem_instance
        )
