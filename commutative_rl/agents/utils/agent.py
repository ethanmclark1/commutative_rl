import os
import yaml
import wandb
import numpy as np

from env import Env

from .helpers import *


class Agent:
    def __init__(
        self,
        seed: int,
        num_instances: int,
        sum_range: range,
        elem_range: range,
        n_elems: int,
        max_noise: float,
    ) -> None:

        self.name = self.__class__.__name__

        cwd = os.getcwd()
        filepath = os.path.join(cwd, "commutative_rl", "agents", "utils", "config.yaml")
        self._init_hyperparams(seed, filepath)

        self.n_elems = n_elems

        self.action_rng = np.random.default_rng(seed)

        self.env = Env(
            seed,
            num_instances,
            sum_range,
            elem_range,
            n_elems,
            max_noise,
            self.config["env"],
        )

        self.q_table = np.zeros(
            (sum_range.stop + 2 * elem_range.stop + max_noise, n_elems)
        )

    def _init_hyperparams(self, seed: int, filepath: str) -> None:
        with open(filepath, "r") as file:
            self.config = yaml.safe_load(file)

        self.seed = seed

        for key, value in self.config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    setattr(self, k, v)
            else:
                setattr(self, key, value)

    def _setup_wandb(self, problem_instance: str) -> None:
        wandb.init(
            project="Set Optimizer",
            entity="ethanmclark1",
            name=f"{self.name} Q-Table",
            tags=[f"{problem_instance.capitalize()}"],
        )

        for key, value in self.config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    wandb.config[k] = v
            else:
                wandb.config[key] = value

    def _setup_problem(self, problem_instance: str) -> None:
        self.env.set_problem(problem_instance)
        self._setup_wandb(problem_instance)

    def _select_action(self, state: np.ndarray, is_eval: bool = False) -> int:
        if is_eval or self.action_rng.random() > self.epsilon:
            sum = int(state[0] * self.env.target_sum)
            action_idx = argmax(self.q_table[sum, :], self.action_rng)
        else:
            action_idx = self.action_rng.integers(self.n_elems)

        return action_idx

    def _update(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        prev_state: np.ndarray = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        state = int(state[0] * self.env.target_sum)
        next_state = int(next_state[0] * self.env.target_sum)

        max_next_q_value = np.max(self.q_table[next_state, :])

        current_q_value = self.q_table[state, action_idx]
        next_q_value = reward + self.gamma * (1 - done) * max_next_q_value

        self.q_table[state, action_idx] += self.alpha * (next_q_value - current_q_value)

    def _train(self) -> None:
        state, done = self.env.reset()

        prev_state = None
        prev_action_idx = None
        prev_reward = None

        timestep = 0
        episode_step = 0
        while timestep < self.n_timesteps:
            action_idx = self._select_action(state)
            next_state, reward, done = self.env.step(state, action_idx, episode_step)

            self._update(
                state,
                action_idx,
                reward,
                next_state,
                done,
                prev_state,
                prev_action_idx,
                prev_reward,
            )

            if done:
                prev_state = None
                prev_action_idx = None
                prev_reward = None

                state, done = self.env.reset()
                episode_step = 0
            else:
                prev_state = state
                prev_action_idx = action_idx
                prev_reward = reward

                state = next_state
                episode_step += 1

            timestep += 1

    def _test(self) -> float:
        returns = []

        for _ in range(self.num_episodes_testing):
            discount = 1.0
            episode_reward = 0.0

            state, done = self.env.reset()
            episode_step = 0

            while not done:
                action_idx = self._select_action(state, is_eval=True)
                next_state, reward, done = self.env.step(
                    state, action_idx, episode_step
                )

                episode_reward += reward * discount
                discount *= self.gamma

                state = next_state
                episode_step += 1

            returns.append(episode_reward)

        return returns

    def generate_target_sum(self, problem_instance: str) -> None:
        self._setup_problem(problem_instance)

        current_n_steps = 0
        for _ in range(self.num_episodes):
            self._train()
            returns = self._test()

            current_n_steps += self.n_timesteps
            avg_returns = np.mean(returns)
            print(f"Average Return: {avg_returns}")

            step = (
                current_n_steps // 3
                if self.name == "TripleTraditional"
                else current_n_steps
            )

            wandb.log({"Average Return": avg_returns}, step=step)

        wandb.finish()
