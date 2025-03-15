import itertools

from agents.traditional import QTable, TripleDataQTable
from agents.commutative import DoubleTableQTable, CombinedRewardQTable, HashMapQTable

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
    ) = get_arguments(n_instances, remaining_argv)

    approach_map = {
        "QTable": QTable,
        "TripleDataQTable": TripleDataQTable,
        "DoubleTableQTable": DoubleTableQTable,
        "CombinedRewardQTable": CombinedRewardQTable,
        "HashMapQTable": HashMapQTable,
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
        )
        for name in approaches
    ]

    for approach, problem_instance in itertools.product(approaches, problem_instances):
        approach.generate_target_sum(problem_instance)
