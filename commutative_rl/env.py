import os
import yaml
import numpy as np

from problems.problem_generator import generate_random_problems


class Env:
    def __init__(
        self,
        seed: int,
        num_instances: int,
        sum_range: range,
        elems_range: range,
        n_elems: int,
        name: str,
        config: dict,
    ) -> None:

        self.target_sum = None
        self.sum_limit = None
        self.elements = None
        self.element_costs = None

        self.cost_rng = np.random.default_rng(seed)
        self.noise_rng = np.random.default_rng(seed)
        self.problem_rng = np.random.default_rng(seed)

        self.num_instances = num_instances
        self.sum_range = sum_range
        self.elems_range = elems_range
        self.n_elems = n_elems
        self.approach_type = "qtable" if "QTable" in name else "dqn"

        self.max_noise = config["max_noise"]
        self.step_value = config["step_value"]
        self.over_penalty = config["over_penalty"]
        self.n_episode_steps = config["n_episode_steps"]

    def set_problem(
        self, problem_instance: str, filename: str = "problems.yaml"
    ) -> None:
        cwd = os.getcwd()
        filepath = os.path.join(cwd, "commutative_rl", "problems", filename)

        while True:
            try:
                with open(filepath, "r") as file:
                    data = yaml.safe_load(file)

                params = data[self.approach_type].get("parameters", {})
                if (
                    params.get("sum_range")
                    == [self.sum_range.start, self.sum_range.stop]
                    and params.get("elems_range")
                    == [self.elems_range[0], self.elems_range.stop]
                    and params.get("n_elems") == self.n_elems
                    and params.get("num_instances") == self.num_instances
                ):
                    problems = data[self.approach_type].get("instances", {})
                    break
                else:
                    raise FileNotFoundError

            except FileNotFoundError:
                generate_random_problems(
                    self.problem_rng,
                    self.sum_range,
                    self.elems_range,
                    self.n_elems,
                    self.num_instances,
                    filepath,
                )

        problem = problems.get(problem_instance)
        self.target_sum = problem.get("target_sum")
        self.sum_limit = self.target_sum * 2.0
        self.elements = problem.get("elements")
        self.element_costs = problem.get("element_costs")
        self.terminal_reward = self.elems_range.stop * self.step_value * 2

    def _get_next_state(self, state: float, action_idx: int) -> float:
        new_elem = self.elements[action_idx]

        if self.approach_type == "qtable":
            noise = self.noise_rng.integers(0, self.max_noise + 1)
        else:
            noise = self.noise_rng.random() * self.max_noise

        noisy_new_elem = (new_elem + noise) / self.target_sum
        next_state = state + noisy_new_elem

        return next_state

    def _get_reward(
        self,
        state: float,
        action_idx: int,
        next_state: float,
    ) -> float:

        if self.elements[action_idx] != 0:
            util_s = self._calc_utility(state)
            util_s_prime = self._calc_utility(next_state)
            reward = util_s_prime - util_s - self.element_costs[action_idx]
        else:
            reward = self.terminal_reward

        return reward

    def _calc_utility(self, state: float) -> float:
        denormalized_state = state * self.target_sum

        if denormalized_state < self.target_sum:
            utility = denormalized_state * self.step_value
        elif denormalized_state < self.sum_limit:
            utility = (
                self.target_sum * self.step_value
                - (denormalized_state - self.target_sum) * self.over_penalty
            )
        else:
            utility = (
                self.target_sum * self.step_value
                - (self.sum_limit - self.target_sum) * self.over_penalty
            )

        return utility

    def step(self, state: float, action_idx: int) -> tuple:
        truncated = self.episode_step >= self.n_episode_steps
        terminated = self.elements[action_idx] == 0

        next_state = state

        if not terminated:
            next_state = self._get_next_state(state, action_idx)
            terminated = next_state >= 2.0  # normalized sum limit

        reward = self._get_reward(state, action_idx, next_state)

        self.episode_step += 1

        return next_state, reward, truncated, terminated

    def reset(self):
        state = 0
        truncated = False
        terminated = False

        self.episode_step = 0

        return state, truncated, terminated
