import itertools

from agents.traditional import QTable, TripleDataQTable
from agents.commutative import (
    DoubleTableQTable,
    CombinedRewardQTable,
    HashMapQTable,
)

from arguments import parse_n_instances, get_arguments


if __name__ == "__main__":
    n_instances, remaining_argv = parse_n_instances()
    (
        seed,
        approaches,
        problem_instances,
        grid_dims,
        n_starts,
        n_goals,
        n_bridges,
        n_episode_steps,
        action_success_rate,
        utility_scale,
        terminal_reward,
        duplicate_bridge_penalty,
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
            grid_dims,
            n_starts,
            n_goals,
            n_bridges,
            n_episode_steps,
            action_success_rate,
            utility_scale,
            terminal_reward,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
        )
        for name in approaches
    ]
    city_reconfiguration = {
        approach.name: {
            problem_instance: None for problem_instance in problem_instances
        }
        for approach in approaches
    }

    for approach, problem_instance in itertools.product(approaches, problem_instances):
        city_reconfiguration[approach.name][
            problem_instance
        ] = approach.generate_city_design(problem_instance)
