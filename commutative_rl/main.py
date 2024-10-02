import os
import yaml
import itertools

from arguments import get_arguments

# from planners.speaker import Speaker
# from planners.listener import Listener

from agents.traditional import Traditional
from agents.commutative import Commutative, CommutativeWithoutIndices
from commutative_rl.agents.triple_data import TripleData

from agents.baselines.grid_world import GridWorld
from agents.baselines.voronoi_map import VoronoiMap
from agents.baselines.direct_path import DirectPath


if __name__ == "__main__":
    cwd = os.getcwd()
    config_path = os.path.join(cwd, "commutative_rl", "agents", "utils", "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    approach_map = {
        "Traditional": Traditional,
        "Commutative": Commutative,
        "CommutativeWithoutIndices": CommutativeWithoutIndices,
        "TripleData": TripleData,
        "GridWorld": GridWorld,
        "VoronoiMap": VoronoiMap,
        "DirectPath": DirectPath,
    }

    (
        num_agents,
        num_large_obstacles,
        num_small_obstacles,
        seed,
        approaches,
        problem_instances,
    ) = get_arguments()

    approaches += ["GridWorld", "VoronoiMap", "DirectPath"]

    approaches = [
        approach_map[name](
            seed, num_agents, num_large_obstacles, num_small_obstacles, config
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
        learned_set[approach.name][problem_instance] = approach.generate_language(
            problem_instance
        )
