import os
import copy
import wandb
import pickle
import numpy as np
import networkx as nx
import gymnasium as gym
import problems.problem_generator as problems


class Env:
    def __init__(
        self,
        seed: int,
        map_size: str,
        num_instances: int,
        reward_type: str,
        noise_type: str,
    ) -> None:

        self.env = gym.make(id="FrozenLake-v1", map_name=map_size, render_mode=None)

        self.seed = seed
        self.reward_type = reward_type
        self.noise_type = noise_type

        self.action_rng = np.random.default_rng(seed)
        self.bridge_rng = np.random.default_rng(seed)
        self.instance_rng = np.random.default_rng(seed)

        n_cols = (
            self.env.unwrapped.ncol if self.env.spec.kwargs["map_name"] == "8x8" else 5
        )
        self.grid_dims = (
            self.env.unwrapped.desc.shape
            if self.env.spec.kwargs["map_name"] == "8x8"
            else (n_cols, n_cols)
        )
        self.problem_size = (
            self.env.spec.kwargs["map_name"]
            if self.env.spec.kwargs["map_name"] == "8x8"
            else "5x5"
        )

        self.state_dims = self.grid_dims[0] * self.grid_dims[1]
        self.num_bridges = 16 if self.problem_size == "8x8" else 8
        self.max_elements = 12 if self.problem_size == "8x8" else 6
        self.action_cost = 0.025 if self.problem_size == "8x8" else 0.075
        # Add a dummy action (+1) to terminate the episode
        self.action_dims = self.num_bridges + 1

        cwd = os.getcwd()
        self.name = self.__class__.__name__
        self.output_dir = f"{cwd}/commutative_rl/history/random_seed={self.seed}"
        os.makedirs(self.output_dir, exist_ok=True)

        problems.generate_instances(
            self.problem_size,
            self.instance_rng,
            self.grid_dims,
            num_instances,
            self.num_bridges,
        )

        # Noise Parameters
        self.configs_to_consider = 1
        self.action_success_rate = 0.50
        self.percent_holes = 0.80 if noise_type == "full" else 1

    def _save(self, problem_instance: str, adaptation: dict) -> None:
        directory = self.output_dir + f"/{self.name.lower()}"
        filename = f"{self.problem_size}_{problem_instance}.pkl"
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "wb") as file:
            pickle.dump(adaptation, file)

    def _load(self, problem_instance: str) -> dict:
        problem_instance = "cheese"
        directory = self.output_dir + f"/{self.name.lower()}"
        filename = f"{self.problem_size}_{problem_instance}.pkl"
        file_path = os.path.join(directory, filename)
        with open(file_path, "rb") as f:
            adaptation = pickle.load(f)
        return adaptation

    def _init_wandb(self, problem_instance: str) -> dict:
        if self.reward_type == "true":
            type_name = f"{self.name} w/ True Reward"
        elif self.reward_type == "approximate":
            type_name = f"{self.name} w/ Approximate Reward"

        wandb.init(
            project=f"{self.problem_size} Frozen Lake",
            entity="ethanmclark1",
            name=f"{type_name}",
            tags=[f"{problem_instance.capitalize()}"],
        )

        config = wandb.config
        return config

    # Initialize action mapping for a given problem instance
    def set_problem(self, problem_instance: str) -> dict:
        self.problem = problems.get_instance(problem_instance)

    def _generate_start_state(self) -> tuple:
        done = False
        num_action = 0
        state = np.zeros(self.grid_dims, dtype=int)

        return state, num_action, done

    def _place_bridge(self, state: np.ndarray, action: int) -> np.ndarray:
        next_state = copy.deepcopy(state)

        if action != 0:
            transformed_action = self.instance["mapping"][action]
            next_state[tuple(transformed_action)] = 1

        return next_state

    def _reassign_states(
        self,
        prev_state: np.ndarray,
        prev_action: int,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> tuple:

        action_a_success = not np.array_equal(prev_state, state)
        action_b_success = not np.array_equal(state, next_state)

        commutative_state = self._place_bridge(prev_state, action)

        if action_a_success and action_b_success:
            pass
        elif not action_a_success and action_b_success:
            if prev_action != action:
                next_state = commutative_state
        elif action_a_success and not action_b_success:
            commutative_state = prev_state
            next_state = state
        else:
            commutative_state = prev_state

        return commutative_state, next_state

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

            start, goal, holes = problems.get_entity_positions(
                self.instance, self.instance_rng, self.percent_holes
            )
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

    def _get_next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        next_state = state.copy()

        if action != 0 and (
            self.action_success_rate == 1
            or self.action_success_rate >= self.action_rng.random()
        ):
            next_state = self._place_bridge(state, action)

        return next_state

    def _get_reward(
        self, state: np.ndarray, action: int, next_state: np.ndarray, num_action: int
    ) -> tuple:
        reward = 0
        done = action == 0
        timeout = num_action == self.max_elements

        if not done:
            if not np.array_equal(state, next_state):
                util_s = self._calc_utility(state)
                util_s_prime = self._calc_utility(next_state)
                reward = util_s_prime - util_s
            reward -= self.action_cost * num_action

        return reward, done or timeout

    def _step(self, state: np.ndarray, action_idx: int, num_action: int) -> tuple:
        action = self.actions[action_idx]
        next_state = self._get_next_state(state, action)
        reward, done = self._get_reward(state, action, next_state, num_action)

        return reward, next_state, done
