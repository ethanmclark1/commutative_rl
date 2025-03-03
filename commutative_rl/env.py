import os
import yaml
import numpy as np
import networkx as nx

from itertools import product
from agents.utils.helpers import random_num_in_range
from problems.problem_generator import generate_random_problems


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
        self.state_dims = self.n_bridges + 1  # configuration value + bridge states
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

        self.bridge_values = self._compute_bridge_values()
        self.total_possible_value = sum(self.bridge_values)

        # Sanity check: all bridges should be valid hole locations
        holes_set = set(self.holes)
        assert all(bridge in holes_set for bridge in self.bridge_locations[:-1])
        assert (
            len(self.bridge_locations) - 1 == self.n_bridges
        ), "Number of bridges does not match bridge locations."

    def _create_base_graph(self) -> nx.Graph:
        graph = nx.grid_graph(dim=self.grid_dims)

        for hole in self.holes:
            if hole in graph:
                graph.remove_node(hole)

        return graph

    def _get_safe_neighbors(self, node: list) -> list:
        x, y = node
        neighbors = []

        for direction_x, direction_y in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + direction_x, y + direction_y
            if (
                0 <= new_x < self.grid_dims[0]
                and 0 <= new_y < self.grid_dims[1]
                and (new_x, new_y) not in self.holes
            ):
                neighbors.append((new_x, new_y))

        return neighbors

    def _compute_bridge_values(self):
        bridge_values = []
        baseline_path_lengths = {}

        base_graph = self._create_base_graph()

        for start, goal in self.path_pairs:
            path = nx.shortest_path(base_graph, start, goal)
            baseline_path_lengths[(start, goal)] = len(path)

        for bridge_idx in range(self.n_bridges):
            bridge_graph = base_graph.copy()
            bridge_loc = self.bridge_locations[bridge_idx]

            bridge_graph.add_node(bridge_loc)

            for neighbor in self._get_safe_neighbors(bridge_loc):
                bridge_graph.add_edge(bridge_loc, neighbor, weight=1)

            value = 0
            for start, goal in self.path_pairs:
                baseline_length = baseline_path_lengths[(start, goal)]
                bridge_path = nx.shortest_path(bridge_graph, start, goal)
                bridge_length = len(bridge_path)

                if bridge_length < baseline_length:
                    improvement = baseline_length - bridge_length
                    value += improvement * self.utility_scale

            bridge_values.append(value / len(self.path_pairs))

        return bridge_values

    def _get_next_state(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        next_state = state.copy()

        if self.action_success_rate > self.action_rng.random():
            next_state[1 + action_idx] = 1.0

            path_value = sum(v * b for v, b in zip(self.bridge_values, next_state[1:]))
            next_state[0] = path_value / self.total_possible_value

        return next_state

    def _calc_utility(self, state: np.ndarray) -> float:
        return int(state[0] * self.total_possible_value)

    def _get_reward(
        self, state: np.ndarray, action_idx: int, next_state: np.ndarray
    ) -> float:

        if self.bridge_locations[action_idx] != 0:
            if state[1 + action_idx] == 1:
                reward = -self.duplicate_bridge_penalty
            else:
                util_s = self._calc_utility(state)
                util_s_prime = self._calc_utility(next_state)
                reward = util_s_prime - util_s - self.bridge_costs[action_idx]
        else:
            reward = self.terminal_reward

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
