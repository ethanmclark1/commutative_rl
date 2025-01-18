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
        config: dict,
    ) -> None:

        self.target_sum = None
        self.elements = None
        self.element_costs = None

        self.cost_rng = np.random.default_rng(seed)
        self.noise_rng = np.random.default_rng(seed)
        self.problem_rng = np.random.default_rng(seed)

        self.num_instances = num_instances
        self.sum_range = sum_range
        self.elems_range = elems_range
        self.n_elems = n_elems

        self.n_steps = config["n_steps"]
        self.max_noise = config["max_noise"]
        self.step_scale = config["step_scale"]
        self.under_penalty = config["under_penalty"]
        self.over_penalty = config["over_penalty"]
        self.completion_reward = config["completion_reward"]

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
        self.element_costs = problem.get("elements_cost")

    def _get_next_state(self, state: float, action_idx: int) -> float:
        new_elem = self.elements[action_idx]

        noise = self.noise_rng.integers(0, self.max_noise)
        noisy_new_elem = new_elem + noise

        new_elem_norm = noisy_new_elem / self.target_sum

        next_state = state + new_elem_norm

        return next_state

    def _get_reward(
        self,
        state: float,
        action_idx: int,
        next_state: float,
        terminated: bool,
    ) -> float:

        reward = next_state - state

        if terminated:
            reward += self.completion_reward
            # 1 represents the target sum
            if next_state > 1:
                reward += (1 - next_state) * self.over_penalty
            else:
                reward += (next_state - 1) * self.under_penalty

        reward -= self.element_costs[action_idx]

        return reward

    def step(self, state: float, action_idx: int, episode_step: int) -> tuple:
        terminated = action_idx == 0
        truncated = episode_step == self.n_steps

        next_state = state

        if not terminated:
            next_state = self._get_next_state(state, action_idx)
            terminated = next_state >= 1

        reward = self._get_reward(state, action_idx, next_state, terminated)

        return next_state, reward, (terminated or truncated)

    def reset(self) -> tuple:
        state = 0.0
        done = False

        return state, done
