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

        self.target_sum = None
        self.elements = None

        self.noise_rng = np.random.default_rng(seed)
        self.problem_rng = np.random.default_rng(seed)

        self.num_instances = num_instances
        self.sum_range = sum_range
        self.elems_range = elems_range
        self.n_elems = n_elems
        self.max_noise = max_noise

        self.n_steps = config["n_steps"]
        self.step_scale = config["step_scale"]
        self.under_penalty = config["under_penalty"]
        self.over_penalty = config["over_penalty"]
        self.complete_reward = config["complete_reward"]
        self.n_statistics = config["n_statistics"]

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
        self.target_sum = problem.get("sum")
        self.elements = problem.get("elements")

    def _get_next_state(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        new_elem = self.elements[action_idx]
        new_elem += self.noise_rng.integers(0, self.max_noise)

        current_sum = int(state[0] * self.target_sum)
        current_n_step = int(state[1] * self.n_steps)

        new_sum = (current_sum + new_elem) / self.target_sum
        new_n_step = (current_n_step + 1) / self.n_steps

        next_state = np.array([new_sum, new_n_step], dtype=float)

        return next_state

    def _get_reward(self, state: int, next_state: int, terminated: bool) -> float:
        current_sum = int(state[0] * self.target_sum)
        next_sum = int(next_state[0] * self.target_sum)

        if terminated:
            if next_sum > self.target_sum:
                reward = (self.target_sum - next_sum) * self.over_penalty
            else:
                reward = (
                    self.complete_reward
                    + (next_sum - self.target_sum) * self.under_penalty
                )
        else:
            reward = (next_sum - current_sum) * self.step_scale

        return reward

    def step(self, state: np.ndarray, action_idx: int, episode_step: int) -> tuple:
        terminated = action_idx == 0
        truncated = episode_step + 1 == self.n_steps

        next_state = state.copy()

        if not terminated:
            next_state = self._get_next_state(state, action_idx)
            next_sum = int(next_state[0] * self.target_sum)
            terminated = next_sum >= self.target_sum

        reward = self._get_reward(state, next_state, terminated)

        return next_state, reward, (terminated or truncated)

    def reset(self) -> tuple:
        state = np.zeros(self.n_statistics, dtype=int)
        done = False

        return state, done
