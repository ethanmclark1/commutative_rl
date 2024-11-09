import os
import copy
import yaml
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
        num_instances: int,
        noise_type: str,
        alpha: float,
        buffer_size: int,
        target_update_freq: int,
    ) -> None:

        self.name = self.__class__.__name__

        cwd = os.getcwd()
        filepath = os.path.join(cwd, "commutative_rl", "agents", "utils", "config.yaml")
        self._init_hyperparams(seed, filepath)

        self.alpha = alpha
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq

        self.config["dqn"]["alpha"] = self.alpha
        self.config["dqn"]["buffer_size"] = self.buffer_size
        self.config["dqn"]["target_update_freq"] = self.target_update_freq

        self.action_rng = np.random.default_rng(seed)

        self.env = Env(
            seed,
            num_instances,
            noise_type,
            self.config["env"],
        )

        self.noise_type = noise_type

        self.n_steps = self.env.n_steps
        self.n_bridges = self.env.n_bridges
        self.n_actions = self.env.n_actions

        self.episode_step_increment = encode(1, self.n_steps)

        self.dqn = DQN(seed, self.n_bridges, self.n_actions, self.alpha)
        self.dqn_target = copy.deepcopy(self.dqn)
        self.replay_buffer = ReplayBuffer(
            seed, self.n_bridges, self.n_steps, self.buffer_size
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
            project="Frozen Lake",
            entity="ethanmclark1",
            name=f"{self.name} DQN w/ Simulated Reward",
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

    def _add_to_buffer(
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
            state, action_idx, reward, next_state, done, episode_step, row
        )

    def _learn(self) -> None:
        if self.batch_size > self.replay_buffer.real_size:
            return

        (
            states,
            action_idxs,
            rewards,
            next_states,
            dones,
            episode_steps,
        ) = self.replay_buffer.sample(self.name, self.batch_size)

        next_episode_steps = episode_steps + self.episode_step_increment

        q_values = self.dqn(states, episode_steps)
        selected_q_values = torch.gather(q_values, 1, action_idxs)

        with torch.no_grad():
            next_q_values = self.dqn_target(next_states, next_episode_steps)
            max_next_q_values = torch.max(next_q_values, dim=1).values.view(-1, 1)
            target_q_values = rewards + self.gamma * ~dones * max_next_q_values

        self.dqn.optimizer.zero_grad()
        loss = self.dqn.loss_fn(selected_q_values, target_q_values)
        loss.backward()
        self.dqn.optimizer.step()

    def _train(self) -> None:
        prev_state = None
        prev_action_idx = None
        prev_reward = None

        state, done = self.env.reset()

        timestep = 0
        episode_step = 0
        while timestep < self.n_timesteps:
            action_idx = self._select_action(state, episode_step)
            next_state, reward, done = self.env.step(state, action_idx, episode_step)

            self._add_to_buffer(
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

                state, done = self.env.reset()
                episode_step = 0
            else:
                prev_state = state
                prev_action_idx = action_idx
                prev_reward = reward

                state = next_state
                episode_step += 1

            if timestep % self.target_update_freq == 0:
                self.dqn_target.load_state_dict(self.dqn.state_dict())

            timestep += 1

    def _test(self) -> float:
        returns = []

        for _ in range(self.num_episodes_testing):
            discount = 1.0
            episode_reward = 0.0

            state, done = self.env.reset()
            episode_step = 0

            while not done:
                action_idx = self._select_action(state, episode_step, is_eval=True)
                next_state, reward, done = self.env.step(
                    state, action_idx, episode_step
                )

                episode_reward += reward * discount
                discount *= self.gamma

                state = next_state
                episode_step += 1

            returns.append(episode_reward)

        return returns

    def generate_city_design(self, problem_instance: str) -> None:
        self._setup_problem(problem_instance)

        current_n_steps = 0
        for _ in range(self.num_episodes):
            self._train()
            returns = self._test()

            current_n_steps += self.n_timesteps

            log_step = (
                current_n_steps // 3
                if self.name == "TripleTraditional"
                else current_n_steps
            )
            avg_returns = np.mean(returns)

            wandb.log({"Average Return": avg_returns}, step=log_step)

        wandb.finish()
