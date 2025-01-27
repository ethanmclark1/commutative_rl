import itertools

from agents.traditional import (
    TraditionalQTable,
    TraditionalDQN,
    TripleTraditionalQTable,
    TripleTraditionalDQN,
)
from agents.commutative import (
    CommutativeQTable,
    CommutativeDQN,
)

from arguments import parse_num_instances, get_arguments


if __name__ == "__main__":
    num_instances, remaining_argv = parse_num_instances()
    (
        seed,
        approaches,
        min_sum_range,
        max_sum_range,
        min_elem_range,
        max_elem_range,
        n_elems,
        problem_instances,
        max_noise,
        alpha,
        epsilon,
        gamma,
        batch_size,
        buffer_size,
        hidden_dims,
        n_hidden_layers,
        target_update_freq,
        dropout,
        step_value,
        over_penalty,
    ) = get_arguments(num_instances, remaining_argv)

    approach_map = {
        "TraditionalQTable": TraditionalQTable,
        "TraditionalDQN": TraditionalDQN,
        "CommutativeQTable": CommutativeQTable,
        "CommutativeDQN": CommutativeDQN,
        "TripleTraditionalQTable": TripleTraditionalQTable,
        "TripleTraditionalDQN": TripleTraditionalDQN,
    }

    approaches = [
        approach_map[name](
            seed,
            num_instances,
            range(min_sum_range, max_sum_range),
            range(min_elem_range, max_elem_range),
            n_elems,
            max_noise,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
            step_value,
            over_penalty,
        )
        for name in approaches
    ]

    for approach, problem_instance in itertools.product(approaches, problem_instances):
        approach.generate_target_sum(problem_instance)
