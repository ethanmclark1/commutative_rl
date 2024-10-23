import os
import yaml
import copy
import numpy as np
import networkx as nx
import gymnasium as gym

from problems.problem_generator import generate_random_problems


class Env:
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
        config: dict,
    ) -> None:

        self.starts = None
        self.goals = None
        self.hole_probs = None
        self.action_map = None

        self.env = gym.make("FrozenLake-v1", map_name=config["map_name"])
        self.grid_dims = self.env.unwrapped.desc.shape

        self.action_rng = np.random.default_rng(seed)
        self.bridge_rng = np.random.default_rng(seed)
        self.problem_rng = np.random.default_rng(seed)

        self.num_instances = num_instances
        self.noise_type = noise_type

        self.n_bridges = config["n_bridges"]
        self.n_steps = config["n_steps"]

        self.action_cost = config["action_cost"]
        self.configs_to_consider = config["configs_to_consider"]
        self.action_success_rate = (
            config["action_success_rate"] if noise_type == "full" else 1
        )

        self.n_states = 2**self.n_bridges
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
                    and params.get("n_bridges") == self.n_bridges
                    and params.get("num_instances") == self.num_instances
                ):
                    problems = data.get("instances", {})
                    break
                else:
                    raise FileNotFoundError

            except FileNotFoundError:
                generate_random_problems(
                    self.problem_rng,
                    self.grid_dims,
                    self.n_bridges,
                    self.num_instances,
                    filepath,
                )

        problem = problems.get(problem_instance)
        self.starts = (
            problem.get("starts")
            if self.noise_type in ["residents", "full"]
            else [problem.get("starts")[0]]
        )
        self.goals = (
            problem.get("goals")
            if self.noise_type in ["residents", "full"]
            else [problem.get("goals")[0]]
        )
        self.holes = problem.get("holes")
        self.hole_probs = problem.get("hole_probs")

        action_map = problem.get("mapping")
        self.action_map = {key: tuple(val) for key, val in action_map.items()}
        self.bridge_locations = list(self.action_map.values())

    def place_bridge(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        next_state = copy.deepcopy(state)

        if action_idx != 0:
            action = self.action_map[action_idx]
            next_state[tuple(action)] = 1

        return next_state

    def _get_next_state(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        next_state = state.copy()

        if self.action_success_rate >= self.action_rng.random():
            next_state = self.place_bridge(state, action_idx)

        return next_state

    def _generate_instance(self) -> tuple:
        start = tuple(self.problem_rng.choice(self.starts))
        goal = tuple(self.problem_rng.choice(self.goals))
        holes = [tuple(hole) for hole in self.holes]

        return start, goal, holes

    # Cell Values: {Frozen: 0, Bridge: 1, Start: 2, Goal: 3, Hole: 4}
    def _calc_utility(self, state: np.ndarray) -> float:
        def manhattan_dist(a: tuple, b: tuple) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        graph = nx.grid_graph(dim=self.grid_dims)
        nx.set_edge_attributes(graph, 1, "weight")

        desc = copy.deepcopy(state).reshape(self.grid_dims)

        utilities = []

        for _ in range(self.configs_to_consider):
            tmp_desc = copy.deepcopy(desc)
            tmp_graph = copy.deepcopy(graph)

            start, goal, holes = self._generate_instance()
            tmp_desc[start], tmp_desc[goal] = 2, 3

            # Only place holes if the cell is frozen
            for hole in holes:
                if tmp_desc[hole] == 0:
                    tmp_desc[hole] = 4
                    tmp_graph.remove_node(hole)

            path = nx.astar_path(tmp_graph, start, goal, manhattan_dist, "weight")
            utility = -len(path)
            utilities.append(utility)

        return np.mean(utilities)

    def _get_reward(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        terminated: bool,
        episode_step: int,
    ) -> float:
        reward = 0.0
        util_s_prime = self._calc_utility(next_state)

        if terminated:
            empty_state = np.zeros(self.grid_dims, dtype=int)
            base_util = self._calc_utility(empty_state)
            reward += util_s_prime - base_util - self.action_cost * episode_step
        else:
            if not np.array_equal(state, next_state):
                util_s = self._calc_utility(state)
                reward += util_s_prime - util_s

        return reward

    def step(self, state: np.ndarray, action_idx: int, episode_step: int) -> tuple:
        terminated = action_idx == 0
        truncated = episode_step + 1 == self.n_steps

        next_state = state.copy()

        if not terminated:
            next_state = self._get_next_state(state, action_idx)

        reward = self._get_reward(state, next_state, terminated, episode_step)

        return next_state, reward, (terminated or truncated)

    def reset(self) -> tuple:
        state = np.zeros(self.grid_dims, dtype=int)
        done = False

        return state, done
