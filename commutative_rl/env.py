import signal8
import numpy as np
import networkx as nx
import gymnasium as gym

from statistics import mean
from shapely.geometry import Point
from agents.utils.helpers import *


class Env(gym.Env):
    def __init__(
        self,
        seed: int,
        n_agents: int,
        n_large_obstacles: int,
        n_small_obstacles: int,
        config: dict,
    ) -> None:

        super(Env, self).__init__()

        self.signal8_env = signal8.env(
            num_agents=n_agents,
            num_large_obstacles=n_large_obstacles,
            num_small_obstacles=n_small_obstacles,
            render_mode=None,
            max_cycles=50,
        )

        self.world = self.signal8_env.world
        self.scenario = self.signal8_env.scenario
        self.obstacle_radius = self.world.large_obstacles[0].radius

        self.n_lines = None
        self.candidate_lines = None
        self.candidate_line_costs = None
        self.terminal_reward = None

        self.problem_rng = np.random.default_rng(seed)

        self.granularity = config["granularity"]
        self.n_episode_steps = config["n_episode_steps"]
        self.safe_area_multiplier = config["safe_area_multiplier"]
        self.failed_path_cost = config["failed_path_cost"]
        self.configs_to_consider = config["configs_to_consider"]

        self._generate_candidate_lines()

        self.state_dims = self.n_episode_steps + 1
        self.n_actions = len(self.candidate_lines)

    def _generate_candidate_lines(self) -> None:
        self.candidate_lines = []
        self.candidate_line_costs = []

        granularity = int(self.granularity * 100)

        for i in range(-100 + granularity, 100, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0, i)]  # Vertical lines
            self.candidate_lines += [(0, 0.1, i)]  # Horizontal lines

        for i in range(-200 + granularity, 200, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0.1, i)]  # Left-to-Right diagonal lines
            self.candidate_lines += [(-0.1, 0.1, i)]  # Right-to-Left diagonal lines

        self.candidate_line_costs = [
            random_num_in_range(self.problem_rng, 0, 0.1)
            for _ in range(len(self.candidate_lines))
        ]

        # Terminating action
        self.candidate_lines += [0]
        self.candidate_line_costs += [0]

    def set_problem(self, problem_instance: str) -> None:
        self.problem_instance = problem_instance
        self.n_lines = len(self.candidate_lines) - 1
        self.terminal_reward = 1

    def _get_next_state(self, state: np.ndarray, action_idx: int) -> list:
        denormalized_state = [int(line_idx * self.n_lines) for line_idx in state]

        non_zero = [elem for elem in denormalized_state if elem != 0]
        non_zero.append(action_idx)
        non_zero = sorted(non_zero)

        denormalized_next_state = non_zero + [0] * (self.state_dims - len(non_zero))
        next_state = [line_idx / self.n_lines for line_idx in denormalized_next_state]

        lines = [self.candidate_lines[elem] for elem in non_zero]
        linestring = convert_to_linestring(lines)
        valid_lines = get_intersecting_lines(linestring)
        self.next_regions = create_regions(valid_lines)

        return next_state

    def _generate_instances(self) -> tuple:
        starts = []
        goals = []
        obstacles = []

        for _ in range(self.configs_to_consider):
            start, goal, obs = get_entity_positions(
                self.scenario, self.world, self.problem_rng, self.problem_instance
            )

            starts.append(start)
            goals.append(goal)
            obstacles.append(obs)

        return starts, goals, obstacles

    def _calc_utility(
        self, regions: list, starts: list, goals: list, obstacles: list
    ) -> float:
        def euclidean_dist(a: int, b: int) -> float:
            return regions[a].centroid.distance(regions[b].centroid)

        utilities = []
        for start, goal, _obstacles in zip(starts, goals, obstacles):
            graph, start_region, goal_region = create_instance(
                regions, start, goal, _obstacles
            )

            try:
                # Graph is discretized by regions so we can use A* search on it
                path = nx.astar_path(graph, start_region, goal_region, euclidean_dist)
                safe_area = [regions[idx].area for idx in path]
                utility = self.safe_area_multiplier * mean(safe_area)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                utility = self.failed_path_cost
            utilities.append(utility)

        avg_utility = np.mean(utilities)

        return avg_utility

    def _get_reward(self, action_idx: int) -> tuple:
        if self.candidate_lines[action_idx] != 0:
            starts, goals, obstacles = self._generate_instances()
            util_s = self._calc_utility(self.regions, starts, goals, obstacles)
            util_s_prime = self._calc_utility(
                self.next_regions, starts, goals, obstacles
            )
            reward = util_s_prime - util_s - self.candidate_line_costs[action_idx]
        else:
            reward = self.terminal_reward

        return reward

    def step(self, state: np.ndarray, action_idx: int) -> tuple:
        terminated = self.candidate_lines[action_idx] == 0
        truncated = self.episode_step >= self.n_episode_steps

        next_state = state.copy()

        if not terminated:
            next_state = self._get_next_state(state, action_idx)

        reward = self._get_reward(action_idx)

        self.episode_step += 1

        return next_state, reward, terminated, truncated

    def reset(self) -> tuple:
        state = np.zeros(self.state_dims)
        terminated = False
        truncated = False

        self.episode_step = 0
        self.regions = [SQUARE]
        self.next_regions = [SQUARE]

        return state, terminated, truncated
