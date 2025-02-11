import itertools

from arguments import get_arguments

# from planners.speaker import Speaker
# from planners.listener import Listener

from agents.traditional import TraditionalDQN
from agents.commutative import CommutativeDQN

from agents.baselines.grid_world import GridWorld
from agents.baselines.voronoi_map import VoronoiMap
from agents.baselines.direct_path import DirectPath


if __name__ == "__main__":
    (
        seed,
        n_agents,
        n_large_obstacles,
        n_small_obstacles,
        approaches,
        problem_instances,
        n_episode_steps,
        configs_to_consider,
        alpha,
        epsilon,
        gamma,
        batch_size,
        buffer_size,
        hidden_dims,
        n_hidden_layers,
        target_update_freq,
        dropout,
    ) = get_arguments()

    approach_map = {
        "TraditionalDQN": TraditionalDQN,
        "CommutativeDQN": CommutativeDQN,
        # "GridWorld": GridWorld,
        # "VoronoiMap": VoronoiMap,
        # "DirectPath": DirectPath,
    }

    approaches = [
        approach_map[name](
            seed,
            n_agents,
            n_large_obstacles,
            n_small_obstacles,
            n_episode_steps,
            configs_to_consider,
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
    # approaches += ["GridWorld", "VoronoiMap", "DirectPath"]

    learned_set = {
        approach.name: {
            problem_instance: None for problem_instance in problem_instances
        }
        for approach in approaches
    }

    for approach, problem_instance in itertools.product(approaches, problem_instances):
        learned_set[approach.name][problem_instance] = approach.generate_language(
            problem_instance
        )
