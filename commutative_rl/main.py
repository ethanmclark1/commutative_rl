import itertools

from agents.traditional import (
    QTable,
    OfflineDQN,
    OnlineDQN,
    TripleDataQTable,
    OnlineTripleDataDQN,
    OfflineTripleDataDQN,
)
from agents.commutative import (
    SuperActionQTable,
    OnlineSuperActionDQN,
    OfflineSuperActionDQN,
    CombinedRewardQTable,
    OnlineCombinedRewardDQN,
    OfflineCombinedRewardDQN,
    HashMapQTable,
    OnlineHashMapDQN,
    OfflineHashMapDQN,
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
        early_termination_penalty,
        duplicate_bridge_penalty,
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
        "OnlineDQN": OnlineDQN,
        "OfflineDQN": OfflineDQN,
        "TripleDataQTable": TripleDataQTable,
        "OnlineTripleDataDQN": OnlineTripleDataDQN,
        "OfflineTripleDataDQN": OfflineTripleDataDQN,
        "SuperActionQTable": SuperActionQTable,
        "OnlineSuperActionDQN": OnlineSuperActionDQN,
        "OfflineSuperActionDQN": OfflineSuperActionDQN,
        "CombinedRewardQTable": CombinedRewardQTable,
        "OnlineCombinedRewardDQN": OnlineCombinedRewardDQN,
        "OfflineCombinedRewardDQN": OfflineCombinedRewardDQN,
        "HashMapQTable": HashMapQTable,
        "OnlineHashMapDQN": OnlineHashMapDQN,
        "OfflineHashMapDQN": OfflineHashMapDQN,
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
            early_termination_penalty,
            duplicate_bridge_penalty,
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
