import os
import yaml
import wandb
import numpy as np

from problems.problem_generator import generate_random_problems


class Env:
    def __init__(
        self,
        seed: int,
        num_instances: int,
        min_dist_bounds: int,
        action_dims: int,
        negative_actions: bool,
        duplicate_actions: bool,
        reward_type: str,
        reward_noise: float,
    ) -> None:

        self.ub_sum = 0
        self.lb_sum = 0
        self.actions = []
        self.target_sum = 0

        self.max_elements = 10
        self.action_cost = 0.01
        self.name = self.__class__.__name__

        self.min_dist_bounds = min_dist_bounds
        self.action_dims = action_dims
        self.reward_type = reward_type
        self.reward_noise = reward_noise
        self.num_instances = num_instances
        self.negative_actions = negative_actions
        self.duplicate_actions = duplicate_actions
        self.target_sum_rng = np.random.default_rng(seed)
        self.reward_noise_rng = np.random.default_rng(seed)

    def _init_wandb(self, problem_instance: str) -> dict:
        if self.reward_type == "true":
            type_name = f"{self.name} w/ True Reward"
        elif self.reward_type == "approximate":
            type_name = f"{self.name} w/ Approximate Reward"

        wandb.init(
            project="Set Optimizer",
            entity="ethanmclark1",
            name=f"{type_name}",
            tags=[f"{problem_instance.capitalize()}"],
        )

        config = wandb.config
        return config

    def _set_problem(
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
                    params.get("min_dist_bounds") == self.min_dist_bounds
                    and params.get("action_dims") == self.action_dims
                    and params.get("num_instances") == self.num_instances
                    and params.get("negative_actions") == self.negative_actions
                    and params.get("duplicate_actions") == self.duplicate_actions
                ):
                    problems = data.get("instances", {})
                    break
                else:
                    raise FileNotFoundError

            except FileNotFoundError:
                generate_random_problems(
                    self.target_sum_rng,
                    self.min_dist_bounds,
                    self.action_dims,
                    self.negative_actions,
                    self.duplicate_actions,
                    self.num_instances,
                    filepath,
                )

        problem = problems.get(problem_instance)
        self.ub_sum = problem.get("ub")
        self.lb_sum = problem.get("lb")
        self.target_sum = problem.get("sum")
        self.actions = problem.get("actions")

    def _generate_start_state(self) -> tuple:
        done = False
        num_action = 0
        state = [0] * self.max_elements

        return state, num_action, done

    def _get_next_state(self, state: list, action: int) -> tuple:
        if action == 0:
            return state

        non_zero_elements = [elem for elem in state if elem != 0]
        non_zero_elements.append(action)
        next_state = sorted(non_zero_elements) + [0] * (
            self.max_elements - len(non_zero_elements)
        )

        return next_state

    def _calc_utility(self, state: list) -> float:
        summed_state = sum(state)

        if self.lb_sum <= summed_state <= self.ub_sum:
            distance = (
                (summed_state - self.lb_sum) / (self.target_sum - self.lb_sum)
                if summed_state <= self.target_sum
                else 1
                + (summed_state - self.target_sum) / (self.ub_sum - self.target_sum)
            )
            utility = 1 - abs(distance - 1)
        else:
            utility = 0.0

        return utility

    def _get_reward(
        self, state: list, action: int, next_state: list, num_action: int
    ) -> tuple:
        reward = 0
        done = action == 0
        timeout = num_action == self.max_elements

        if not done:
            util_s = self._calc_utility(state)
            util_s_prime = self._calc_utility(next_state)
            reward = util_s_prime - util_s - self.action_cost * num_action

        reward = self.reward_noise_rng.normal(reward, self.reward_noise)
        return reward, done or timeout

    def _step(self, state: list, action_idx: int, num_action: int) -> tuple:
        action = self.actions[action_idx]
        next_state = self._get_next_state(state, action)
        reward, done = self._get_reward(state, action, next_state, num_action)

        return reward, next_state, done
