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
        max_noise: float,
        config: dict,
    ) -> None:

        self.sum = None
        self.elements = None

        self.problem_rng = np.random.default_rng(seed)
        self.max_noise_rng = np.random.default_rng(seed)

        self.num_instances = num_instances
        self.sum_range = sum_range
        self.elems_range = elems_range
        self.n_elems = n_elems
        self.max_noise = max_noise

        self.n_steps = config["n_steps"]
        self.under_penalty = config["under_penalty"]
        self.over_penalty = config["over_penalty"]
        self.complete_reward = config["complete_reward"]

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
                    params.get("sum_range") == [self.sum_range[0], self.sum_range[-1]]
                    and params.get("elems_range")
                    == [self.elems_range[0], self.elems_range[-1]]
                    and params.get("n_elems") == self.n_elems
                    and params.get("num_instances") == self.num_instances
                ):
                    problems = data.get("instances", {})
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
        self.sum = problem.get("sum")
        self.elements = problem.get("elements")

        non_zero_elements = [elem for elem in self.elements if elem != 0]
        self.min_reward = (min(non_zero_elements) - self.sum) * self.under_penalty

    def _get_next_state(self, state: int, action_idx: int) -> int:
        next_state = state

        next_state += self.elements[action_idx]
        next_state += self.max_noise_rng.integers(0, self.max_noise)

        return next_state

    def _get_reward(self, state: int, next_state: int, terminated: bool) -> float:
        reward = 0.0
        util_s = state

        if terminated:
            if next_state > self.sum:
                reward += (self.sum - next_state) * self.over_penalty
            else:
                reward += self.complete_reward
                reward += (next_state - self.sum) * self.under_penalty
        else:
            util_s_prime = next_state
            reward += util_s_prime - util_s

        return reward

    def step(self, state: int, action_idx: int) -> tuple:
        self.step_count += 1

        terminated = action_idx == 0
        truncated = self.step_count == self.n_steps

        next_state = state

        if not terminated:
            next_state = self._get_next_state(state, action_idx)
            terminated = next_state >= self.sum

        reward = self._get_reward(state, next_state, terminated)

        return next_state, reward, terminated, truncated

    def reset(self) -> tuple:
        self.step_count = 0

        state = 0
        terminated = False
        truncated = False

        return state, terminated, truncated
