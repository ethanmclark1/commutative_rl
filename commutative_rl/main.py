import itertools

from agents.traditional import TraditionalDQN, TripleTraditionalDQN
from agents.commutative import CommutativeDQN

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
        configs_to_consider,
        action_success_rate,
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
        "TraditionalDQN": TraditionalDQN,
        "CommutativeDQN": CommutativeDQN,
        "TripleTraditionalDQN": TripleTraditionalDQN,
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
            configs_to_consider,
            action_success_rate,
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
