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
        self.max_steps = config["max_steps"]

        self.action_cost = config["action_cost"]
        self.configs_to_consider = config["configs_to_consider"]
        self.percent_holes = config["percent_holes"] if noise_type == "full" else 1
        self.action_success_rate = (
            config["action_success_rate"] if noise_type != "none" else 1
        )

        self.n_states = 2**self.n_bridges
        self.n_actions = self.n_bridges + 1
        self.n_cells = np.prod(self.grid_dims)

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
        self.starts = problem.get("starts")
        self.goals = problem.get("goals")
        self.holes = problem.get("holes")
        self.hole_probs = problem.get("hole_probs")

        action_map = problem.get("mapping")
        self.action_map = {key: tuple(val) for key, val in action_map.items()}

    def convert_to_state(self, state_idx: int) -> np.ndarray:
        state = np.zeros(self.grid_dims, dtype=int)
        binary_str = format(state_idx, f"0{self.action_map}b")
        binary_str = binary_str[::-1]

        for action, (row, col) in self.action_map.items():
            state[row, col] = int(binary_str[action])

    def convert_to_idx(self, state: np.ndarray) -> int:
        tmp_state = [state[row, col] for row, col in self.action_map.values()]
        binary_arr = tmp_state[::-1]
        binary_str = "".join(map(str, binary_arr))
        state_idx = int(binary_str, 2)

        return state_idx

    def _place_bridge(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        next_state = copy.deepcopy(state)

        if action_idx != 0:
            action = self.action_map[action_idx]
            next_state[tuple(action)] = 1

        return next_state

    def _get_next_state(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        next_state = state.copy()

        if action_idx != 0 and (
            self.action_success_rate == 1
            or self.action_success_rate >= self.action_rng.random()
        ):
            next_state = self._place_bridge(state, action_idx)

        return next_state

    def _generate_instance(self) -> tuple:
        start = tuple(self.problem_rng.choice(self.starts))
        goal = tuple(self.problem_rng.choice(self.goals))

        holes = self.holes

        if self.percent_holes != 1:
            num_holes = int(len(self.holes) * self.percent_holes)
            normalized_probs = np.array(self.hole_probs) / np.sum(self.hole_probs)
            holes = self.problem_rng.choice(self.holes, num_holes, p=normalized_probs)

        holes = [tuple(hole) for hole in holes]

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
        episode_step: int,
        terminated: bool,
    ) -> float:
        reward = 0.0

        util_s_prime = self._calc_utility(next_state)

        if not terminated:
            if not np.array_equal(state, next_state):
                util_s = self._calc_utility(state)
                reward += util_s_prime - util_s

        if terminated:
            reward += util_s_prime
            reward -= self.action_cost * episode_step

        return reward

    def step(self, state: np.ndarray, action_idx: int, episode_step: int) -> tuple:
        terminated = False
        truncated = episode_step >= self.max_steps

        if action_idx == 0:
            terminated = True
            next_state = state
        else:
            next_state = self._get_next_state(state, action_idx)

        reward = self._get_reward(state, next_state, episode_step, terminated)

        return next_state, reward, terminated, truncated

    def reset(self) -> tuple:
        state = np.zeros(self.grid_dims, dtype=int)
        terminated = False
        truncated = False

        return state, terminated, truncated
