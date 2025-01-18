import os
import yaml
import copy
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
        sum_range: range,
        elem_range: range,
        n_elems: int,
        max_noise: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        learning_start_step: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        activation_fn: str = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        grad_clip_norm: float = None,
        loss_fn: str = None,
        layer_norm: int = None,
        step_scale: float = None,
        over_penalty: float = None,
        under_penalty: float = None,
        completion_reward: float = None,
    ) -> None:

        self.name = self.__class__.__name__

        cwd = os.getcwd()
        filepath = os.path.join(cwd, "commutative_rl", "agents", "utils", "config.yaml")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._init_params(
            filepath,
            seed,
            n_elems,
            max_noise,
            alpha,
            epsilon,
            gamma,
            batch_size,
            learning_start_step,
            buffer_size,
            hidden_dims,
            activation_fn,
            n_hidden_layers,
            target_update_freq,
            grad_clip_norm,
            loss_fn,
            bool(layer_norm) if layer_norm is not None else None,
            step_scale,
            over_penalty,
            under_penalty,
            completion_reward,
        )

        self.env = Env(
            seed,
            num_instances,
            sum_range,
            elem_range,
            n_elems,
            self.config["env"],
        )

        if "QTable" in self.name:
            self.q_table = np.zeros(
                (sum_range.stop + 2 * elem_range.stop + self.env.max_noise, n_elems)
            )
        else:
            self.network = DQN(
                seed,
                1,
                n_elems,
                self.hidden_dims,
                self.activation_fn,
                self.n_hidden_layers,
                self.loss_fn,
                self.alpha,
                self.layer_norm,
            ).to(self.device)
            self.target_network = copy.deepcopy(self.network)
            self.buffer = ReplayBuffer(
                seed,
                self.batch_size,
                self.buffer_size,
                self.device,
            )

        self.action_rng = np.random.default_rng(seed)

        self.best_model_state = None
        self.best_avg_return = -np.inf

    def _init_params(
        self,
        filepath: str,
        seed: int,
        n_elems: int,
        max_noise: float = None,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        learning_start_step: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        activation_fn: str = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        grad_clip_norm: float = None,
        loss_fn: str = None,
        layer_norm: bool = None,
        step_scale: float = None,
        over_penalty: float = None,
        under_penalty: float = None,
        completion_reward: float = None,
    ) -> None:
        with open(filepath, "r") as file:
            self.config = yaml.safe_load(file)

        self.seed = seed
        self.n_elems = n_elems

        approach = "qtable" if "QTable" in self.name else "dqn"
        non_approach = "dqn" if "QTable" in self.name else "qtable"

        # Override default environment values with command line arguments
        if step_scale is not None:
            self.config["env"]["step_scale"] = step_scale
        if over_penalty is not None:
            self.config["env"]["over_penalty"] = over_penalty
        if under_penalty is not None:
            self.config["env"]["under_penalty"] = under_penalty
        if completion_reward is not None:
            self.config["env"]["completion_reward"] = completion_reward
        if max_noise is not None:
            self.config["env"]["max_noise"] = max_noise

        # Override default agent values with command line arguments
        if alpha is not None:
            self.config["agent"][approach]["alpha"] = alpha
        if epsilon is not None:
            self.config["agent"][approach]["epsilon"] = epsilon
        if gamma is not None:
            self.config["agent"][approach]["gamma"] = gamma
        if batch_size is not None:
            self.config["agent"]["dqn"]["batch_size"] = batch_size
        if learning_start_step is not None:
            self.config["agent"]["dqn"]["learning_start_step"] = learning_start_step
        if buffer_size is not None:
            self.config["agent"]["dqn"]["buffer_size"] = buffer_size
        if hidden_dims is not None:
            self.config["agent"]["dqn"]["hidden_dims"] = hidden_dims
        if activation_fn is not None:
            self.config["agent"]["dqn"]["activation_fn"] = activation_fn
        if n_hidden_layers is not None:
            self.config["agent"]["dqn"]["n_hidden_layers"] = n_hidden_layers
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
                    else:
                        wandb.config[k] = v
            else:
                wandb.config[key] = value

    def _setup_problem(self, problem_instance: str) -> None:
        self.env.set_problem(problem_instance)
        self._setup_wandb(problem_instance)

        self.update_step = 0

    def _save_model(self, problem_instance: str) -> None:
        cwd = os.getcwd()
        self.ckpt_dir = os.path.join(cwd, "commutative_rl", "ckpt", problem_instance)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        if "QTable" in self.name:
            self.ckpt_path = os.path.join(self.ckpt_dir, f"{self.name}.npy")
            np.save(
                self.ckpt_path,
                self.best_q_table if hasattr(self, "best_q_table") else self.q_table,
            )
        else:
            self.ckpt_path = os.path.join(self.ckpt_dir, f"{self.name}.pt")
            torch.save(
                self.best_model_state
                if self.best_model_state is not None
                else self.network.state_dict(),
                self.ckpt_path,
            )

    def _select_action(self, state: np.ndarray, is_eval: bool = False) -> int:
        if is_eval or self.action_rng.random() > self.epsilon:
            if "QTable" in self.name:
                sum = int(state * self.env.target_sum)
                action_idx = argmax(self.q_table[sum, :], self.action_rng)
            else:
                processed_state = torch.as_tensor(
                    [state], dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    action_idx = self.network(processed_state).argmax().item()
        else:
            action_idx = self.action_rng.integers(self.n_elems)

        return action_idx

    def _update(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        done: bool,
        prev_state: float = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        state = int(state * self.env.target_sum)
        next_state = int(next_state * self.env.target_sum)

        max_next_q_value = np.max(self.q_table[next_state, :])

        current_q_value = self.q_table[state, action_idx]
        next_q_value = reward + self.gamma * (1 - done) * max_next_q_value

        self.q_table[state, action_idx] += self.alpha * (next_q_value - current_q_value)

    def _add_to_buffer(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        done: bool,
        prev_state: float = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        self.buffer.add(state, action_idx, reward, next_state, done)

    def _learn(self, current_n_steps: int, train_step: int) -> None:
        total_step = (
            (current_n_steps + train_step) // 3
            if self.name == "TripleTraditionalDQN"
            else current_n_steps + train_step
        )

        if total_step < self.learning_start_step:
            return

        for i in range(3):
            states, action_idxs, rewards, next_states, dones = self.buffer.sample()

            current_q_values = self.network(states).gather(1, action_idxs)

            with torch.no_grad():
                next_actions = self.network(next_states).argmax(dim=1).view(-1, 1)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
                target_q_values = rewards + self.gamma * ~dones * next_q_values

            self.network.optimizer.zero_grad()
            loss = self.network.loss_fn(current_q_values, target_q_values)
            loss.backward()

            if self.grad_clip_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), max_norm=self.grad_clip_norm
                )

            self.network.optimizer.step()

            self.update_step += 1

            if self.name == "TripleTraditionalDQN":
                break

        if self.update_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def _train(self, current_n_steps: int) -> None:
        state, done = self.env.reset()

        prev_state = None
        prev_action_idx = None
        prev_reward = None

        train_step = 0
        episode_step = 0
        while train_step < self.n_training_steps:
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
                )

            if done:
                if "DQN" in self.name:
                    self._learn(current_n_steps, train_step)

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

            train_step += 1

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

            current_n_steps += self.n_training_steps
            avg_returns = np.mean(returns)

            # Save best model
            if avg_returns > self.best_avg_return:
                self.best_avg_return = avg_returns
                if "QTable" in self.name:
                    self.best_q_table = self.q_table.copy()
                else:
                    self.best_model_state = copy.deepcopy(self.network.state_dict())

            log_step = (
                current_n_steps // 3
                if "TripleTraditional" in self.name
                else current_n_steps
            )

            wandb.log({"Average Return": self.best_avg_return}, step=log_step)

        wandb.finish()

        self._save_model(problem_instance)
