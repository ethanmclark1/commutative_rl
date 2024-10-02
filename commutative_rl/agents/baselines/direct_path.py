import numpy as np

from env import Env
from scipy.spatial import cKDTree
from agents.utils.helpers import *
from agents.baselines.utils.rrt_star import RRTStar


class DirectPath:
    tree = None

    def __init__(
        self,
        seed: int,
        num_agents: int,
        num_large_obstacles: int,
        num_small_obstacles: int,
        config: dict,
    ) -> None:

        self.name = self.__class__.__name__

        self.env = Env(
            seed,
            num_agents,
            num_large_obstacles,
            num_small_obstacles,
            config["env"],
        )

        self.num_configs = 100
        self.world_rng = np.random.default_rng(seed)

        agent_radius = self.env.world.agents[0].radius
        goal_radius = self.env.world.goals[0].radius
        obstacle_radius = self.env.world.small_obstacles[0].radius
        self.planner = RRTStar(agent_radius, goal_radius, obstacle_radius)

    @staticmethod
    def get_point_index(point):
        return DirectPath.tree.query(point)[1]

    def get_language(self, problem_instance: str) -> list:
        obstacles = []
        direct_path = None
        for _ in range(self.num_configs):
            start, goal, obs = get_entity_positions(
                self.env.scenario, self.env.world, self.env.world_rng, problem_instance
            )
            obstacles.extend(obs)

        while direct_path is None:
            direct_path = self.planner.get_path(start, goal, obstacles)
        DirectPath.tree = cKDTree(direct_path)

        return direct_path
