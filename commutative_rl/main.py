import itertools

from agents.traditional import Traditional, TripleTraditional
from agents.commutative import (
    Commutative,
    CommutativeIndependentSamples,
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
        target_update_freq,
        grad_clip_norm,
        loss_fn,
        layer_norm,
        aggregation_type,
    ) = get_arguments(num_instances, remaining_argv)

    approach_map = {
        "Traditional": Traditional,
        "Commutative": Commutative,
        "CommutativeIndependentSamples": CommutativeIndependentSamples,
        "TripleTraditional": TripleTraditional,
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
            target_update_freq,
            grad_clip_norm,
            loss_fn,
            layer_norm,
            aggregation_type,
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
        approach.generate_target_sum(problem_instance)
