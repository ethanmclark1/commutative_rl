import os
import yaml
import numpy as np
import networkx as nx

from agents.utils.helpers import encode, decode
from problems.problem_generator import generate_random_problems

from enum import Enum


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
        name: str,
        config: dict,
    ) -> None:

        self.starts = None
        self.goals = None
        self.holes = None
        self.bridge_locations = None
        self.bridge_costs = None
        self.terminal_reward = None

        self.action_rng = np.random.default_rng(seed)
        self.bridge_rng = np.random.default_rng(seed)
        self.problem_rng = np.random.default_rng(seed)

        self.n_instances = n_instances
        self.approach_type = "qtable" if "QTable" in name else "dqn"

        grid_dim = int(config["grid_dims"].split("x")[0])
        self.grid_dims = (grid_dim, grid_dim)
        self.n_starts = config["n_starts"]
        self.n_goals = config["n_goals"]
        self.n_bridges = config["n_bridges"]
        self.n_holes = config["n_holes"]
        self.noise_type = config["noise_type"]
        self.n_episode_steps = config["n_episode_steps"]
        self.configs_to_consider = config["configs_to_consider"]
        self.action_success_rate = config["action_success_rate"]

        self.n_states = 2**self.n_bridges - 1
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
                    and params.get("n_holes") == self.n_holes
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
                    self.n_holes,
                    self.n_instances,
                    filepath,
                )

        problem = problems.get(problem_instance)

        self.starts = (
            problem.get("starts")
            if self.noise_type in ["Residents", "Full"]
            else [problem.get("starts")[0]]
        )
        self.goals = (
            problem.get("goals")
            if self.noise_type in ["Residents", "Full"]
            else [problem.get("goals")[0]]
        )
        self.holes = problem.get("holes")
        self.bridge_locations = problem.get("bridge_locations")
        self.bridge_costs = problem.get("bridge_costs")
        self.terminal_reward = 5

    def _get_next_state(self, state: int | float, action_idx: int) -> int | float:
        state = decode(state, self.n_bridges, self.n_states)

        if self.action_success_rate > self.action_rng.random():
            state[action_idx] = CellValues.Bridge.value

        next_state = encode(state, self.approach_type, self.n_states)

        return next_state

    def _generate_instances(self) -> tuple:
        config_starts = map(
            tuple, self.problem_rng.choice(self.starts, size=self.configs_to_consider)
        )
        config_goals = map(
            tuple, self.problem_rng.choice(self.goals, size=self.configs_to_consider)
        )

        return (list(config_starts), list(config_goals), list(map(tuple, self.holes)))

    def _get_grid_state(self, state: int | float) -> np.ndarray:
        grid_state = np.zeros(self.grid_dims, dtype=int)

        state = decode(state, self.n_bridges, self.n_states)

        for idx, loc in enumerate(state):
            if loc == CellValues.Bridge.value:
                bridge = self.bridge_locations[idx]
                grid_state[tuple(bridge)] = CellValues.Bridge.value

        return grid_state

    def _calc_utility(
        self, state: int | float, starts: list, goals: list, holes: list
    ) -> float:
        def manhattan_dist(a: tuple, b: tuple) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        grid_state = self._get_grid_state(state)

        graph = nx.grid_graph(dim=self.grid_dims)
        nx.set_edge_attributes(graph, 1, "weight")

        for loc in holes:
            if grid_state[loc] == CellValues.Frozen.value:
                grid_state[loc] = CellValues.Hole.value
                graph.remove_node(loc)

        utilities = []
        for start, goal in zip(starts, goals):
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
        state: int | float,
        action_idx: int,
        next_state: int | float,
    ) -> float:

        if self.bridge_locations[action_idx] != 0:
            starts, goals, holes = self._generate_instances()
            util_s = self._calc_utility(state, starts, goals, holes)
            util_s_prime = self._calc_utility(next_state, starts, goals, holes)
            reward = util_s_prime - util_s - self.bridge_costs[action_idx]
        else:
            reward = self.terminal_reward

        return reward

    def step(self, state: int | float, action_idx: int) -> tuple:
        truncated = self.episode_step >= self.n_episode_steps
        terminated = self.bridge_locations[action_idx] == 0

        next_state = state

        if not terminated:
            next_state = self._get_next_state(state, action_idx)

        reward = self._get_reward(state, action_idx, next_state)

        self.episode_step += 1

        return next_state, reward, truncated, terminated

    def reset(self):
        state = 0
        truncated = False
        terminated = False

        self.episode_step = 0

        return state, truncated, terminated
