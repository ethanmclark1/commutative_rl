import os
import yaml
import numpy as np
import networkx as nx

from enum import Enum
from itertools import product
from problems.problem_generator import generate_random_problems


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
        self.holes = None
        self.bridge_locations = None
        self.bridge_costs = None
        self.terminal_reward = None

        self.action_rng = np.random.default_rng(seed)
        self.problem_rng = np.random.default_rng(seed)

        self.n_instances = n_instances

        grid_dim = int(config["grid_dims"].split("x")[0])
        self.grid_dims = (grid_dim, grid_dim)
        self.n_starts = config["n_starts"]
        self.n_goals = config["n_goals"]
        self.n_bridges = config["n_bridges"]
        self.n_episode_steps = config["n_episode_steps"]
        self.action_success_rate = config["action_success_rate"]

        self.state_dims = self.n_bridges
        self.n_actions = self.n_bridges + 1

    def set_problem(
        self, problem_instance: str, filename: str = "problems.yaml"
    ) -> None:
        cwd = os.getcwd()
        filepath = os.path.join(cwd, "commutative_rl", "problems", filename)

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
                    self.problem_rng,
                    self.grid_dims,
                    self.n_starts,
                    self.n_goals,
                    self.n_bridges,
                    self.n_instances,
                    filepath,
                )

        problem = problems.get(problem_instance)

        self.path_pairs = [
            [tuple(start), tuple(goal)]
            for start, goal in product(problem.get("starts"), problem.get("goals"))
        ]
        self.holes = [tuple(hole) for hole in problem.get("holes")]
        self.bridge_locations = problem.get("bridge_locations")
        self.bridge_costs = problem.get("bridge_costs")
        self.terminal_reward = 0.50

    def _get_next_state(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        next_state = state.copy()

        if self.action_success_rate > self.action_rng.random():
            next_state[action_idx] = CellValues.Bridge.value

        return next_state

    def _create_instance(self, state: np.ndarray) -> np.ndarray:
        grid_state = np.zeros(self.grid_dims, dtype=int)
        graph = nx.grid_graph(dim=self.grid_dims)
        nx.set_edge_attributes(graph, 1, "weight")

        for idx, loc in enumerate(state):
            if loc == CellValues.Bridge.value:
                bridge = self.bridge_locations[idx]
                grid_state[tuple(bridge)] = CellValues.Bridge.value

        for loc in self.holes:
            if grid_state[loc] == CellValues.Frozen.value:
                grid_state[loc] = CellValues.Hole.value
                graph.remove_node(loc)

        return grid_state, graph

    def _calc_utility(self, state: np.ndarray) -> float:
        def manhattan_dist(a: tuple, b: tuple) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        grid_state, graph = self._create_instance(state)

        utilities = []
        for start, goal in self.path_pairs:
            tmp_grid = grid_state.copy()
            tmp_grid[start] = CellValues.Start.value
            tmp_grid[goal] = CellValues.Goal.value

            path = nx.astar_path(graph, start, goal, manhattan_dist)
            utility = -len(path)
            utilities.append(utility)

        avg_utility = np.mean(utilities)

        return avg_utility

    def _get_reward(
        self,
        state: np.ndarray,
        action_idx: int,
        next_state: np.ndarray,
    ) -> float:

        if self.bridge_locations[action_idx] != 0:
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
        state = np.zeros(self.state_dims, dtype=int)
        terminated = False
        truncated = False

        self.episode_step = 0

        return state, terminated, truncated
