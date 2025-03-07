import os
import yaml
import numpy as np
import networkx as nx

from enum import Enum
from itertools import product
from problems.problem_generator import generate_random_problems
from agents.utils.helpers import random_num_in_range, visualize_grid


class CellValues(Enum):
    Frozen = 0
    Bridge = 1
    Start = 2
    Goal = 3
    Hole = 4


class Env:
    def __init__(
        self,
        seed: int,
        n_instances: int,
        config: dict,
    ) -> None:

        self.path_pairs = None
        self.starts = None
        self.goals = None
        self.holes = None
        self.bridge_locations = None
        self.bridge_values = None
        self.bridge_costs = None
        self.total_possible_value = None

        self.action_rng = np.random.default_rng(seed)
        self.problem_rng = np.random.default_rng(seed)

        self.n_instances = n_instances

        grid_dim = int(config["grid_dims"].split("x")[0])
        self.n_starts = config["n_starts"]
        self.n_goals = config["n_goals"]
        self.n_bridges = config["n_bridges"]
        self.n_episode_steps = config["n_episode_steps"]
        self.action_success_rate = config["action_success_rate"]
        self.utility_scale = config["utility_scale"]
        self.terminal_reward = config["terminal_reward"]
        self.bridge_cost_lb = config["bridge_cost_lb"]
        self.bridge_cost_ub = config["bridge_cost_ub"]
        self.duplicate_bridge_penalty = config["duplicate_bridge_penalty"]

        self.grid_dims = (grid_dim, grid_dim)
        self.state_dims = self.n_bridges + 1  # mean shortest path + bridge states
        self.n_actions = self.n_bridges + 1  # bridges + terminal action

    def get_problem(self, problem_instance: str) -> None:
        cwd = os.getcwd()
        filepath = os.path.join(cwd, "commutative_rl", "problems", "problems.yaml")

        while True:
            try:
                with open(filepath, "r") as file:
                    data = yaml.safe_load(file)

                params = data.get("parameters", {})
                if (
                    params.get("grid_dims") == list(self.grid_dims)
                    and params.get("n_starts") == self.n_starts
                    and params.get("n_goals") == self.n_goals
                    and params.get("n_bridges") == self.n_bridges
                    and params.get("n_instances") == self.n_instances
                ):
                    problems = data.get("instances", {})
                    break
                else:
                    raise FileNotFoundError

            except FileNotFoundError:
                generate_random_problems(
                    self.n_instances,
                    self.grid_dims,
                    self.problem_rng,
                    self.n_starts,
                    self.n_goals,
                    self.n_bridges,
                    filepath,
                )
        problem = problems.get(problem_instance)

        self.starts = [tuple(start) for start in problem.get("starts")]
        self.goals = [tuple(goal) for goal in problem.get("goals")]
        self.path_pairs = [
            [start, goal] for start, goal in product(self.starts, self.goals)
        ]
        self.holes = [tuple(hole) for hole in problem.get("holes")]
        self.bridge_locations = [
            tuple(bridge) for bridge in problem.get("bridge_locations")
        ]

        self.bridge_costs = [
            random_num_in_range(
                self.problem_rng, self.bridge_cost_lb, self.bridge_cost_ub
            )
            for _ in range(self.n_bridges)
        ]
        self.bridge_locations.append(0)  # Add terminal action

        # Sanity check: all bridges should be valid hole locations
        holes_set = set(self.holes)
        assert all(bridge in holes_set for bridge in self.bridge_locations[:-1])
        assert (
            len(self.bridge_locations) - 1 == self.n_bridges
        ), "Number of bridges does not match bridge locations."

        self.shortest_path_lengths = self.get_shortest_path_lengths()
        self.longest_path_lengths = self.get_longest_path_lengths()

    def _create_instance(self, state: np.ndarray) -> nx.Graph:
        grid_state = np.zeros(self.grid_dims, dtype=int)
        graph = nx.grid_graph(dim=self.grid_dims)

        for idx, loc in enumerate(state[1:]):
            if loc == CellValues.Bridge.value:
                bridge = self.bridge_locations[idx]
                grid_state[tuple(bridge)] = CellValues.Bridge.value

        for loc in self.holes:
            if grid_state[loc] == CellValues.Frozen.value:
                grid_state[loc] = CellValues.Hole.value
                graph.remove_node(loc)

        return graph

    # Find shortest path lengths between start-goal pairs when all bridges are present
    def get_shortest_path_lengths(self):
        graph = self._create_instance(np.ones(self.state_dims))

        path_lengths = []
        for start, goal in self.path_pairs:
            path_len = len(nx.shortest_path(graph, start, goal))
            path_lengths.append(path_len)
        avg_shortest_path_len = np.mean(path_lengths)

        return avg_shortest_path_len

    # Find longest path lengths between start-goal pairs when all bridges are present
    def get_longest_path_lengths(self):
        graph = self._create_instance(np.zeros(self.state_dims))

        path_lengths = []
        for start, goal in self.path_pairs:
            path_len = len(nx.shortest_path(graph, start, goal))
            path_lengths.append(path_len)
        avg_longest_path_len = np.mean(path_lengths)

        return avg_longest_path_len

    def _get_next_state(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        next_state = state.copy()

        if self.action_success_rate > self.action_rng.random():
            next_state[1 + action_idx] = CellValues.Bridge.value

            graph = self._create_instance(next_state)

            path_lengths = []
            for start, goal in self.path_pairs:
                path_len = len(nx.shortest_path(graph, start, goal))
                path_lengths.append(path_len)
            avg_path_length = np.mean(path_lengths)

            normalized_value = 1.0 - (avg_path_length - self.shortest_path_lengths) / (
                self.longest_path_lengths - self.shortest_path_lengths
            )
            next_state[0] = normalized_value

        return next_state

    def _calc_utility(self, state: np.ndarray) -> float:
        return state[0]

    def _get_reward(
        self, state: np.ndarray, action_idx: int, next_state: np.ndarray
    ) -> float:

        if self.bridge_locations[action_idx] != 0:
            if state[1 + action_idx] == 1:
                reward = -self.duplicate_bridge_penalty
            else:
                util_s = self._calc_utility(state)
                util_s_prime = self._calc_utility(next_state)
                reward = (
                    self.utility_scale * (util_s_prime - util_s)
                    - self.bridge_costs[action_idx]
                )
        else:
            reward = self.terminal_reward * next_state[0]

        return reward

    def step(self, state: np.ndarray, action_idx: int) -> tuple:
        terminated = self.bridge_locations[action_idx] == 0
        truncated = self.episode_step >= self.n_episode_steps

        next_state = state.copy()

        if not terminated:
            next_state = self._get_next_state(state, action_idx)

        reward = self._get_reward(state, action_idx, next_state)

        self.episode_step += 1

        return next_state, reward, terminated, truncated

    def reset(self):
        state = np.zeros(self.state_dims, dtype=float)
        terminated = False
        truncated = False

        self.episode_step = 0

        return state, terminated, truncated
