import os
import yaml
import numpy as np
import networkx as nx

from enum import Enum
from itertools import product
from problems.problem_generator import generate_random_problems
from agents.utils.helpers import random_num_in_range, encode, decode, visualize_grid


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
        self.bridge_costs = None

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
        self.early_termination_penalty = config["early_termination_penalty"]
        self.bridge_cost_lb = config["bridge_cost_lb"]
        self.bridge_cost_ub = config["bridge_cost_ub"]
        self.duplicate_bridge_penalty = config["duplicate_bridge_penalty"]
        self.bridge_stages = config["bridge_stages"]

        self.grid_dims = (grid_dim, grid_dim)
        self.state_dims = 1
        # bridge_stages + 2 to allow for overbuilt bridges
        self.n_states = (self.bridge_stages + 2) ** self.n_bridges
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

        self.bridge_cost_distribution = [
            {
                "mean": self.problem_rng.uniform(
                    self.bridge_cost_lb, self.bridge_cost_ub
                ),
                "std": self.problem_rng.uniform(
                    self.bridge_cost_lb, self.bridge_cost_ub
                ),
            }
            for _ in range(self.n_bridges)
        ]

        self.bridge_locations.append(0)  # Add terminal action

        # Sanity check: all bridges should be valid hole locations
        holes_set = set(self.holes)
        assert all(bridge in holes_set for bridge in self.bridge_locations[:-1])
        assert (
            len(self.bridge_locations) - 1 == self.n_bridges
        ), "Number of bridges does not match bridge locations."

        self.shortest_path_lengths = self.get_path_lengths(
            self.bridge_stages * np.ones(self.n_bridges, dtype=int)
        )  # All bridges
        self.longest_path_lengths = self.get_path_lengths(
            np.zeros(self.n_bridges, dtype=int)
        )  # No bridges

        self._path_length_cache = {}

    def _create_instance(self, bridge_progress: np.ndarray) -> nx.Graph:
        grid_state = np.zeros(self.grid_dims, dtype=np.float16)
        graph = nx.grid_graph(dim=self.grid_dims)
        nx.set_edge_attributes(graph, 1, "weight")

        for idx, completion in enumerate(bridge_progress):
            if self.bridge_locations[idx] != 0:
                bridge = self.bridge_locations[idx]

                if completion == self.bridge_stages:
                    grid_state[bridge] = CellValues.Bridge.value
                elif 0 < completion < self.bridge_stages:
                    # For visualization
                    grid_state[bridge] = CellValues.Bridge.value * (
                        completion / self.bridge_stages
                    )

                    # Partially built bridges have higher traversal cost
                    for neighbor in graph.neighbors(bridge):
                        graph[bridge][neighbor]["weight"] = 5

        for loc in self.holes:
            if grid_state[loc] == CellValues.Frozen.value:
                grid_state[loc] = CellValues.Hole.value
                graph.remove_node(loc)

        return graph

    def get_path_lengths(self, bridge_progression: np.ndarray) -> float:
        graph = self._create_instance(bridge_progression)

        path_lengths = []
        for start, goal in self.path_pairs:
            path_len = nx.astar_path_length(graph, start, goal)
            path_lengths.append(path_len)
        avg_path_length = np.mean(path_lengths)

        return avg_path_length

    def _get_next_state(self, decoded_state: float, action_idx: int) -> np.ndarray:
        decoded_next_state = decoded_state.copy()

        if self.action_success_rate > self.action_rng.random():
            decoded_next_state[action_idx] += 1

        return decoded_next_state

    def _calc_utility(self, decoded_state: np.ndarray) -> float:
        state_key = tuple(decoded_state)
        if state_key in self._path_length_cache:
            avg_path_length = self._path_length_cache[state_key]
        else:
            avg_path_length = self.get_path_lengths(decoded_state)
            self._path_length_cache[state_key] = avg_path_length

        path_efficiency = 1.0 - (avg_path_length - self.shortest_path_lengths) / (
            self.longest_path_lengths - self.shortest_path_lengths
        )
        return path_efficiency

    def _get_reward(
        self, decoded_state: np.ndarray, action_idx: int, decoded_next_state: np.ndarray
    ) -> float:

        util_s_prime = self._calc_utility(decoded_next_state)
        if self.bridge_locations[action_idx] == 0:
            # Only reward if 50% progress is made to optimal policy
            if util_s_prime > 0.5:
                reward = self.terminal_reward * util_s_prime
            else:
                reward = -self.early_termination_penalty
        # Penalize if bridge is overbuilt
        elif decoded_next_state[action_idx] > self.bridge_stages:
            reward = -self.duplicate_bridge_penalty
        else:
            util_s = self._calc_utility(decoded_state)
            utility_improvement = self.utility_scale * (util_s_prime - util_s)
            bridge_cost = abs(
                self.problem_rng.normal(
                    self.bridge_cost_distribution[action_idx]["mean"],
                    self.bridge_cost_distribution[action_idx]["std"],
                )
            )

            reward = utility_improvement - bridge_cost

        return reward

    def step(self, state: float, action_idx: int) -> tuple:
        terminated = self.bridge_locations[action_idx] == 0
        truncated = self.episode_step == self.n_episode_steps

        decoded_state = decode(state, self.bridge_stages, self.n_bridges, self.n_states)
        decoded_next_state = decoded_state.copy()

        if not terminated:
            decoded_next_state = self._get_next_state(decoded_state, action_idx)

        next_state = encode(
            decoded_next_state,
            self.bridge_stages,
            self.n_bridges,
            self.n_states,
        )
        reward = self._get_reward(decoded_state, action_idx, decoded_next_state)

        self.episode_step += 1

        return next_state, reward, terminated, truncated

    def reset(self) -> float:
        state = 0.0
        terminated = False
        truncated = False

        self.episode_step = 0
        self.bridge_progress = np.zeros(self.n_bridges, dtype=int)

        return state, terminated, truncated
