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
        learning_start_step,
        buffer_size,
        hidden_dims,
        activation_fn,
        n_hidden_layers,
        target_update_freq,
        grad_clip_norm,
        loss_fn,
        layer_norm,
        step_scale,
        over_penalty,
        under_penalty,
        completion_reward,
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
            learning_start_step,
            buffer_size,
            hidden_dims,
            activation_fn,
            n_hidden_layers,
            target_update_freq,
            grad_clip_norm,
            loss_fn,
            layer_norm,
            step_scale,
            over_penalty,
            under_penalty,
            completion_reward,
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
