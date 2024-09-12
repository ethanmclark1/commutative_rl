import os
import yaml
import wandb
import numpy as np

from env import Env
from .helpers import *


class Parent:
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
    ) -> None:

        self.name = self.__class__.__name__

        cwd = os.getcwd()
        filepath = os.path.join(cwd, "commutative_rl", "agents", "utils", "config.yaml")
        self._init_hyperparams(seed, filepath)

        self.action_rng = np.random.default_rng(seed)
        self.hallucination_rng = np.random.default_rng(seed)

        self.env = Env(
            seed,
            num_instances,
            noise_type,
            self.config["env"],
        )

        self.n_states = self.env.n_states
        self.n_actions = self.env.n_actions

        self.q_table = np.zeros((self.n_states, self.n_actions))

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
            project="Frozen Lake",
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

    def _select_action(
        self, state: int, episode_step: int, is_eval: bool = False
    ) -> int:
        if is_eval or self.action_rng.random() > self.epsilon:
            state_idx = self.env.convert_to_idx(state)
            action_idx = argmax(self.q_table[state_idx, :], self.action_rng)
        else:
            action_idx = self.action_rng.integers(self.n_actions)

        return action_idx

    def _update(
        self,
        state: int,
        action_idx: int,
        reward: float,
        next_state: int,
        terminated: bool,
        prev_state: int = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        state_idx = self.env.convert_to_idx(state)
        next_state_idx = self.env.convert_to_idx(next_state)
        max_next_state = np.max(self.q_table[next_state_idx, :])
        self.q_table[state_idx, action_idx] += self.alpha * (
            reward + self.gamma * max_next_state - self.q_table[state_idx, action_idx]
        )

    def _train(self) -> None:
        prev_state = None
        prev_action_idx = None
        prev_reward = None

        state, terminated, truncated = self.env.reset()

        episode_step = 0
        training_step = 0
        while training_step < self.n_timesteps:
            action_idx = self._select_action(state, episode_step)
            next_state, reward, terminated, truncated = self.env.step(
                state, action_idx, episode_step
            )

            self._update(
                state,
                action_idx,
                reward,
                next_state,
                terminated,
                prev_state,
                prev_action_idx,
                prev_reward,
            )

            if terminated or truncated:
                prev_state = None
                prev_action_idx = None
                prev_reward = None

                state, terminated, truncated = self.env.reset()

                episode_step = 0
            else:
                prev_state = state
                prev_action_idx = action_idx
                prev_reward = reward

                state = next_state
                episode_step += 1

            training_step += 1

    def _test(self) -> float:
        returns = []

        for _ in range(self.num_episodes_testing):
            discount = 1.0
            episode_reward = 0.0

            state, terminated, truncated = self.env.reset()

            while not (terminated or truncated):
                action_idx = self._select_action(state, is_eval=True)
                next_state, reward, terminated, truncated = self.env.step(
                    state, action_idx
                )

                state = next_state

                episode_reward += reward * discount
                discount *= self.gamma

            returns.append(episode_reward)

        return returns

    def generate_city_design(self, problem_instance: str) -> None:
        self.env.set_problem(problem_instance)
        # self._setup_wandb(problem_instance)

        current_n_steps = 0
        for _ in range(self.num_episodes):
            self._train()
            returns = self._test()

            current_n_steps += self.n_timesteps
            avg_returns = np.mean(returns)

            wandb.log({"Average Return": avg_returns}, step=current_n_steps)

        wandb.finish()
