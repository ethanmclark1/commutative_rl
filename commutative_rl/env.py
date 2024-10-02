import signal8
import warnings
import numpy as np
import networkx as nx
import gymnasium as gym

from statistics import mean
from shapely.geometry import Point
from agents.utils.helpers import *

warnings.filterwarnings("ignore", message="invalid value encountered in intersection")


class Env(gym.Env):
    def __init__(
        self,
        seed: int,
        num_agents: int,
        num_large_obstacles: int,
        num_small_obstacles: int,
        config: dict,
    ) -> None:
        super(Env, self).__init__()

        self.lines = []

        self.signal8_env = signal8.env(
            num_agents=num_agents,
            num_large_obstacles=num_large_obstacles,
            num_small_obstacles=num_small_obstacles,
            render_mode=None,
            max_cycles=50,
        )

        self.world = self.signal8_env.world
        self.scenario = self.signal8_env.scenario
        self.obstacle_radius = self.world.large_obstacles[0].radius

        self.world_rng = np.random.default_rng(seed)

        self.n_steps = config["n_steps"]
        self.granularity = config["granularity"]

        self.action_cost = config["action_cost"]
        self.util_multiplier = config["util_multiplier"]
        self.failed_path_cost = config["failed_path_cost"]
        self.configs_to_consider = config["configs_to_consider"]

        self._generate_action_set()

        self.n_actions = len(self.lines) + 1

    def _generate_action_set(self) -> None:
        granularity = int(self.granularity * 100)

        for i in range(-100 + granularity, 100, granularity):
            i /= 1000
            self.lines += [(0.1, 0, i)]  # Vertical lines
            self.lines += [(0, 0.1, i)]  # Horizontal lines

        for i in range(-200 + granularity, 200, granularity):
            i /= 1000
            self.lines += [(0.1, 0.1, i)]  # Left-to-Right diagonal lines
            self.lines += [(-0.1, 0.1, i)]  # Right-to-Left diagonal lines

    def set_problem(self, problem_instance: str) -> None:
        self.problem_instance = problem_instance

    def _get_next_state(self, state: list, action_idx: int) -> tuple:
        if action_idx == 0:
            return state

        next_state = [elem for elem in state if elem != 0]
        next_state.append(action_idx)
        next_state = sorted(next_state)
        next_state = next_state + [0] * (self.n_steps - len(next_state))

        return next_state

    def _get_regions(self, state: list) -> list:
        non_zero = [elem for elem in state if elem != 0]
        lines = [self.lines[elem - 1] for elem in non_zero]

        linestring = convert_to_linestring(lines)
        valid_lines = get_intersecting_lines(linestring)
        next_regions = create_regions(valid_lines)

        return next_regions

    def _calc_utility(self, regions: list) -> float:
        def euclidean_dist(a: int, b: int) -> float:
            return regions[a].centroid.distance(regions[b].centroid)

        utilities = []

        for _ in range(self.configs_to_consider):
            start, goal, obstacles = get_entity_positions(
                self.scenario, self.world, self.world_rng, self.problem_instance
            )
            obstacles_with_size = [
                Point(obs_pos).buffer(self.obstacle_radius) for obs_pos in obstacles
            ]
            graph, start_region, goal_region = create_instance(
                regions, start, goal, obstacles_with_size
            )

            try:
                path = nx.astar_path(graph, start_region, goal_region, euclidean_dist)
                safe_area = [regions[idx].area for idx in path]
                utility = self.util_multiplier * mean(safe_area)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                utility = self.failed_path_cost

            utilities.append(utility)

        return np.mean(utilities)

    def _get_reward(
        self,
        regions: list,
        next_regions: list,
        terminated: bool,
        episode_step: int,
    ) -> tuple:
        reward = 0.0
        util_s = self._calc_utility(regions)

        if terminated:
            empty_region = SQUARE
            base_utility = self._calc_utility([empty_region])
            reward += util_s - base_utility
        else:
            if len(regions) != len(next_regions):
                util_s_prime = self._calc_utility(next_regions)
                reward += util_s_prime - util_s - (episode_step * self.action_cost)

        return reward

    def step(
        self, state: np.ndarray, regions: list, action_idx: int, episode_step: int
    ) -> tuple:
        terminated = action_idx == 0
        truncated = episode_step + 1 == self.n_steps

        next_state = state.copy()
        next_regions = regions.copy()

        if not terminated:
            next_state = self._get_next_state(state, action_idx)
            next_regions = self._get_regions(next_state)

        reward = self._get_reward(regions, next_regions, terminated, episode_step + 1)

        return next_state, next_regions, reward, (terminated or truncated)

    def reset(self) -> tuple:
        state = np.full(self.n_steps, 0)
        regions = [SQUARE]
        done = False

        return state, regions, done
