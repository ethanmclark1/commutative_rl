import itertools

from agents.traditional import QTable, DQN, TripleDataQTable, TripleDataDQN
from agents.commutative import (
    SuperActionQTable,
    SuperActionDQN,
    CombinedRewardQTable,
    CombinedRewardDQN,
    HashMapQTable,
    HashMapDQN,
)

from arguments import parse_n_instances, get_arguments


if __name__ == "__main__":
    n_instances, remaining_argv = parse_n_instances()
    (
        seed,
        approaches,
        min_sum_range,
        max_sum_range,
        min_elem_range,
        max_elem_range,
        n_actions,
        problem_instances,
        max_noise,
        step_value,
        over_penalty,
        alpha,
        epsilon,
        gamma,
        batch_size,
        buffer_size,
        hidden_dims,
        n_hidden_layers,
        target_update_freq,
        dropout,
    ) = get_arguments(n_instances, remaining_argv)

    approach_map = {
        "QTable": QTable,
        "DQN": DQN,
        "TripleDataQTable": TripleDataQTable,
        "TripleDataDQN": TripleDataDQN,
        "SuperActionQTable": SuperActionQTable,
        "SuperActionDQN": SuperActionDQN,
        "CombinedRewardQTable": CombinedRewardQTable,
        "CombinedRewardDQN": CombinedRewardDQN,
        "HashMapQTable": HashMapQTable,
        "HashMapDQN": HashMapDQN,
    }

    approaches = [
        approach_map[name](
            seed,
            n_instances,
            range(min_sum_range, max_sum_range),
            range(min_elem_range, max_elem_range),
            n_actions,
            max_noise,
            step_value,
            over_penalty,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
        )
        for name in approaches
    ]

    for approach, problem_instance in itertools.product(approaches, problem_instances):
        approach.generate_target_sum(problem_instance)
