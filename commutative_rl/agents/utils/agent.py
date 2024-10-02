import copy
import torch
import wandb
import numpy as np

from env import Env

from .helpers import *
from .networks import DQN
from .buffers import ReplayBuffer


class Agent:
    def __init__(
        self,
        seed: int,
        num_agents: int,
        num_large_obstacles: int,
        num_small_obstacles: int,
        config: dict,
    ) -> None:

        self.name = self.__class__.__name__

        self._init_hyperparams(seed, config)

        self.action_rng = np.random.default_rng(seed)
        self.hallucination_rng = np.random.default_rng(seed)

        self.env = Env(
            seed,
            num_agents,
            num_large_obstacles,
            num_small_obstacles,
            config["env"],
        )

        self.n_steps = self.env.n_steps
        self.n_actions = self.env.n_actions

        self.episode_step_increment = encode(1, self.n_steps)

        self.dqn = DQN(seed, self.n_steps, self.n_actions, self.alpha)
        self.dqn_target = copy.deepcopy(self.dqn)
        self.replay_buffer = ReplayBuffer(
            seed, self.n_steps, self.n_actions, self.buffer_size
        )

    def _init_hyperparams(self, seed: int, config: dict) -> None:
        self.seed = seed
        self.config = config

        for key, value in self.config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    setattr(self, k, v)
            else:
                setattr(self, key, value)

    def _setup_wandb(self, problem_instance: str) -> None:
        wandb.init(
            project="Signal8",
            entity="ethanmclark1",
            name=f"{self.name} DQN",
            tags=[f"{problem_instance.capitalize()}"],
        )

        for key, value in self.config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    wandb.config[k] = v
            else:
                wandb.config[key] = value

        wandb.config["noise_type"] = self.noise_type

    def _setup_problem(self, problem_instance: str) -> None:
        self.env.set_problem(problem_instance)
        self._setup_wandb(problem_instance)

    def _select_action(
        self, state: int, episode_step: int, is_eval: bool = False
    ) -> int:
        if is_eval or self.action_rng.random() > self.epsilon:
            episode_step = encode(episode_step, self.n_steps, to_tensor=True)
            state = torch.as_tensor(state, dtype=torch.float32)

            with torch.no_grad():
                action_idx = self.dqn(state, episode_step).argmax().item()
        else:
            action_idx = self.action_rng.integers(self.n_actions)

        return action_idx

    def _add_to_buffers(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        episode_step: int,
        row: int = 0,
    ) -> None:

        self.replay_buffer.add(
            state,
            action_idx,
            reward,
            next_state,
            done,
            episode_step,
            row,
        )

    def _learn(self) -> None:
        if replay_buffer is None or self.batch_size > replay_buffer.real_size:
            return None

        if indices is None:
            indices = replay_buffer.sample(self.batch_size)

        states = self.replay_buffer.states[indices]
        action_idxs = self.replay_buffer.action_idxs[indices]
        rewards = self.replay_buffer.rewards[indices]
        next_states = self.replay_buffer.next_states[indices]
        dones = self.replay_buffer.dones[indices]
        episode_steps = self.replay_buffer.episode_steps[indices]
        next_episode_steps = episode_steps + self.episode_step_increment

        q_values = self.dqn(states, episode_steps)
        selected_q_values = torch.gather(q_values, 1, action_idxs)

        with torch.no_grad():
            next_q_values = self.dqn_target(next_states, next_episode_steps)
            max_next_q_values = torch.max(next_q_values, dim=1).values.view(-1, 1)
            target_q_values = rewards + self.gamma * ~dones * max_next_q_values

        self.dqn.optimizer.zero_grad()
        loss = self.dqn.loss(selected_q_values, target_q_values)
        loss.backward()
        self.dqn.optimizer.step()

    def _train(self) -> None:
        prev_state = None
        prev_action_idx = None
        prev_reward = None

        state, regions, done = self.env.reset()

        timestep = 0
        episode_step = 0
        while timestep < self.n_timesteps:
            action_idx = self._select_action(state, episode_step)
            next_state, next_regions, reward, done = self.env.step(
                state, regions, action_idx, episode_step
            )

            self._add_to_buffers(
                state,
                action_idx,
                reward,
                next_state,
                done,
                episode_step,
                prev_state,
                prev_action_idx,
                prev_reward,
            )

            if done:
                self._learn()

                prev_state = None
                prev_action_idx = None
                prev_reward = None

                state, regions, done = self.env.reset()
                episode_step = 0
            else:
                prev_state = state
                prev_action_idx = action_idx
                prev_reward = reward

                state = next_state
                regions = next_regions
                episode_step += 1

            if timestep % self.target_update_freq == 0:
                self.dqn_target.load_state_dict(self.dqn.state_dict())

            timestep += 1

    def _test(self) -> float:
        returns = []

        for _ in range(self.num_episodes_testing):
            discount = 1.0
            episode_reward = 0.0

            state, regions, done = self.env.reset()
            episode_step = 0

            while not done:
                action_idx = self._select_action(state, episode_step, is_eval=True)
                next_state, next_regions, reward, done = self.env.step(
                    state, regions, action_idx, episode_step
                )

                episode_reward += reward * discount
                discount *= self.gamma

                state = next_state
                regions = next_regions
                episode_step += 1

            returns.append(episode_reward)

        return returns

    def generate_language(self, problem_instance: str) -> None:
        self._setup_problem(problem_instance)

        current_n_steps = 0
        for _ in range(self.num_episodes):
            self._train(current_n_steps)
            returns = self._test()

            current_n_steps += self.n_timesteps
            avg_returns = np.mean(returns)

            step = (
                current_n_steps // 3 if self.name == "TripleData" else current_n_steps
            )

            wandb.log({"Average Return": avg_returns}, step=step)

        wandb.finish()
