import os
import yaml
import copy
import wandb
import numpy as np

from env import Env

from .helpers import *
from .networks import DuelingDQN
from .buffers import ReplayBuffer


class Agent:
    def __init__(
        self,
        seed: int,
        num_instances: int,
        sum_range: range,
        elem_range: range,
        n_elems: int,
        max_noise: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        target_update_freq: int = None,
        grad_clip_norm: float = None,
        loss_fn: str = None,
        layer_norm: int = None,
        aggregation_type: str = None,
    ) -> None:

        self.name = self.__class__.__name__

        cwd = os.getcwd()
        filepath = os.path.join(cwd, "commutative_rl", "agents", "utils", "config.yaml")

        self._init_params(
            filepath,
            seed,
            n_elems,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            target_update_freq,
            grad_clip_norm,
            loss_fn,
            bool(layer_norm) if layer_norm is not None else None,
            aggregation_type,
        )

        self.env = Env(
            seed,
            num_instances,
            sum_range,
            elem_range,
            n_elems,
            max_noise,
            self.config["env"],
        )

        if "QTable" in self.name:
            self.q_table = np.zeros(
                (sum_range.stop + 2 * elem_range.stop + max_noise, n_elems)
            )
        else:
            self.network = DuelingDQN(
                seed,
                self.env.n_statistics,
                n_elems,
                self.hidden_dims,
                self.loss_fn,
                self.alpha,
                self.layer_norm,
            )
            self.target_network = copy.deepcopy(self.network)
            self.buffer = ReplayBuffer(
                seed, self.env.n_statistics, self.batch_size, self.buffer_size
            )

        self.action_rng = np.random.default_rng(seed)

    def _init_params(
        self,
        filepath: str,
        seed: int,
        n_elems: int,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        target_update_freq: int = None,
        grad_clip_norm: float = None,
        loss_fn: str = None,
        layer_norm: bool = None,
        aggregation_type: str = None,
    ) -> None:
        with open(filepath, "r") as file:
            self.config = yaml.safe_load(file)

        self.seed = seed
        self.n_elems = n_elems

        approach = "qtable" if "QTable" in self.name else "dqn"
        non_approach = "dqn" if "QTable" in self.name else "qtable"

        # Override default values with command line arguments
        if alpha is not None:
            self.config["agent"][approach]["alpha"] = alpha
        if epsilon is not None:
            self.config["agent"][approach]["epsilon"] = epsilon
        if gamma is not None:
            self.config["agent"][approach]["gamma"] = gamma
        if aggregation_type is not None:
            self.config["agent"][approach]["aggregation_type"] = aggregation_type
        if batch_size is not None:
            self.config["agent"]["dqn"]["batch_size"] = batch_size
        if buffer_size is not None:
            self.config["agent"]["dqn"]["buffer_size"] = buffer_size
        if hidden_dims is not None:
            self.config["agent"]["dqn"]["hidden_dims"] = hidden_dims
        if target_update_freq is not None:
            self.config["agent"]["dqn"]["target_update_freq"] = target_update_freq
        if grad_clip_norm is not None:
            self.config["agent"]["dqn"]["grad_clip_norm"] = grad_clip_norm
        if loss_fn is not None:
            self.config["agent"]["dqn"]["loss_fn"] = loss_fn
        if layer_norm is not None:
            self.config["agent"]["dqn"]["layer_norm"] = layer_norm

        del self.config["agent"][non_approach]

        for key, value in self.config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            setattr(self, kk, vv)
                    setattr(self, k, v)
            else:
                setattr(self, key, value)

    def _setup_wandb(self, problem_instance: str) -> None:
        wandb.init(
            project="Set Optimizer",
            entity="ethanmclark1",
            name=f"{self.name}",
            tags=[f"{problem_instance.capitalize()}"],
        )

        for key, value in self.config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            wandb.config[kk] = vv
                    wandb.config[k] = v
            else:
                wandb.config[key] = value

    def _setup_problem(self, problem_instance: str) -> None:
        self.env.set_problem(problem_instance)
        self._setup_wandb(problem_instance)

    def _save_model(self, problem_instance: str) -> None:
        cwd = os.getcwd()
        self.ckpt_dir = os.path.join(cwd, "commutative_rl", "ckpt", problem_instance)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        if "QTable" in self.name:
            self.ckpt_path = os.path.join(self.ckpt_dir, f"{self.name}.npy")
            np.save(self.ckpt_path, self.q_table)
        else:
            self.ckpt_path = os.path.join(self.ckpt_dir, f"{self.name}.pt")
            torch.save(self.network.state_dict(), self.ckpt_path)

    def _monitor_progress(
        self,
        current_n_steps: int,
        timestep: int,
        states: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        total_norm = 0
        norms_by_layer = {}

        log_step = (
            (current_n_steps + timestep) // 3
            if "TripleTraditional" in self.name
            else current_n_steps + timestep
        )

        for name, p in self.network.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
                norms_by_layer[f"grad_norm/{name}"] = param_norm

        total_norm = total_norm**0.5

        with torch.no_grad():
            sample_states = states[:32]
            features = self.network.feature(sample_states)

            value = self.network.value(features)
            advantage = self.network.advantage(features)

            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        wandb.log(
            {
                "q_values/mean": q_values.mean().item(),
                "value_stream/mean": value.mean().item(),
                "advantage_stream/mean": advantage.mean().item(),
                "advantage_stream/std": advantage.std().item(),
                "training/loss": loss.item(),
                "gradients/total_norm": total_norm,
                **norms_by_layer,
            },
            step=log_step,
        )

    def _select_action(self, state: np.ndarray, is_eval: bool = False) -> int:
        if is_eval or self.action_rng.random() > self.epsilon:
            if "QTable" in self.name:
                sum = int(state[0] * self.env.target_sum)
                action_idx = argmax(self.q_table[sum, :], self.action_rng)
            else:
                processed_state = torch.as_tensor(state, dtype=torch.float32)

                with torch.no_grad():
                    action_idx = self.network(processed_state).argmax().item()
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
        episode_step: int = None,
    ) -> None:

        state = int(state[0] * self.env.target_sum)
        next_state = int(next_state[0] * self.env.target_sum)

        max_next_q_value = np.max(self.q_table[next_state, :])

        current_q_value = self.q_table[state, action_idx]
        next_q_value = reward + self.gamma * (1 - done) * max_next_q_value

        self.q_table[state, action_idx] += self.alpha * (next_q_value - current_q_value)

    def _add_to_buffer(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        prev_state: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:
        raise NotImplementedError

    def _learn(self, current_n_steps: int, timestep: int) -> None:
        if self.batch_size > self.buffer.real_size:
            return

        states, action_idxs, rewards, next_states, dones = self.buffer.sample(self.name)

        q_values = self.network(states)
        selected_q_values = torch.gather(q_values, 1, action_idxs)

        with torch.no_grad():
            next_actions = self.network(next_states).argmax(dim=1).view(-1, 1)
            next_q_values = self.target_network(next_states)
            max_next_q_values = torch.gather(next_q_values, 1, next_actions)
            target_q_values = rewards + self.gamma * ~dones * max_next_q_values

        self.network.optimizer.zero_grad()
        loss = self.network.loss_fn(selected_q_values, target_q_values)
        loss.backward()

        self._monitor_progress(current_n_steps, timestep, states, loss)

        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), max_norm=self.grad_clip_norm
        )

        self.network.optimizer.step()

    def _train(self, current_n_steps: int) -> None:
        state, done = self.env.reset()

        prev_state = None
        prev_action_idx = None
        prev_reward = None

        timestep = 0
        episode_step = 0
        while timestep < self.n_timesteps:
            action_idx = self._select_action(state)
            next_state, reward, done = self.env.step(state, action_idx, episode_step)

            if "QTable" in self.name:
                self._update(
                    state,
                    action_idx,
                    reward,
                    next_state,
                    done,
                    prev_state,
                    prev_action_idx,
                    prev_reward,
                    episode_step,
                )
            else:
                self._add_to_buffer(
                    state,
                    action_idx,
                    reward,
                    next_state,
                    done,
                    prev_state,
                    prev_action_idx,
                    prev_reward,
                    episode_step,
                )

            if done:
                if "DQN" in self.name:
                    self._learn(current_n_steps, timestep)

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

            if "DQN" in self.name:
                if timestep % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

    def _test(self) -> float:
        returns = []

        for _ in range(self.num_episodes_testing):
            episode_reward = 0.0

            state, done = self.env.reset()
            episode_step = 0

            while not done:
                action_idx = self._select_action(state, is_eval=True)
                next_state, reward, done = self.env.step(
                    state, action_idx, episode_step
                )

                episode_reward += reward

                state = next_state
                episode_step += 1

            returns.append(episode_reward)

        return returns

    def generate_target_sum(self, problem_instance: str) -> None:
        self._setup_problem(problem_instance)

        current_n_steps = 0
        for _ in range(self.num_episodes):
            self._train(current_n_steps)
            returns = self._test()

            current_n_steps += self.n_timesteps
            avg_returns = np.mean(returns)

            log_step = (
                current_n_steps // 3
                if "TripleTraditional" in self.name
                else current_n_steps
            )

            wandb.log({"Average Return": avg_returns}, step=log_step)

        wandb.finish()

        self._save_model(problem_instance)
