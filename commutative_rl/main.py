import itertools

from agents.traditional import Traditional, TripleTraditional
from agents.commutative import (
    Commutative,
    TripleCommutative,
    CommutativeFullBatch,
    CommutativeWithoutIndices,
)
from arguments import parse_num_instances, get_arguments


if __name__ == "__main__":
    num_instances, remaining_argv = parse_num_instances()
    (
        seed,
        approaches,
        problem_instances,
        noise_type,
        alpha,
        buffer_size,
        target_update_freq,
    ) = get_arguments(num_instances, remaining_argv)

    approach_map = {
        "Traditional": Traditional,
        "Commutative": Commutative,
        "TripleTraditional": TripleTraditional,
        "TripleCommutative": TripleCommutative,
        "CommutativeFullBatch": CommutativeFullBatch,
        "CommutativeWithoutIndices": CommutativeWithoutIndices,
    }

    approaches = [
        approach_map[name](
            seed,
            num_instances,
            noise_type,
            alpha,
            buffer_size,
            target_update_freq,
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
        learned_set[approach.name][problem_instance] = approach.generate_city_design(
            problem_instance
        )
