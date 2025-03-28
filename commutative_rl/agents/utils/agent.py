import os
import yaml
import copy
import wandb
import numpy as np

from env import Env

from .helpers import *


class Agent:
    def __init__(
        self,
        seed: int,
        n_instances: int,
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        step_value: float = None,
        over_penalty: float = None,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
    ) -> None:

        self.name = self.__class__.__name__

        cwd = os.getcwd()
        filepath = os.path.join(cwd, "commutative_rl", "agents", "utils", "config.yaml")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._init_params(
            filepath,
            seed,
            max_noise,
            step_value,
            over_penalty,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
        )

        self.env = Env(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
            self.name,
            self.config["env"],
        )

        self.state_dims = self.env.state_dims
        self.n_actions = self.env.n_actions

        self.action_rng = np.random.default_rng(seed)

    def _init_params(
        self,
        filepath: str,
        seed: int,
        max_noise: float = None,
        step_value: float = None,
        over_penalty: float = None,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
    ) -> None:

        with open(filepath, "r") as file:
            self.config = yaml.safe_load(file)

        self.seed = seed

        approach = "qtable" if "QTable" in self.name else "dqn"
        non_approach = "dqn" if "QTable" in self.name else "qtable"

        # Override default environment values with command line arguments
        if step_value is not None:
            self.config["env"]["step_value"] = step_value
        if over_penalty is not None:
            self.config["env"]["over_penalty"] = over_penalty
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
        if buffer_size is not None:
            self.config["agent"]["dqn"]["buffer_size"] = buffer_size
        if hidden_dims is not None:
            self.config["agent"]["dqn"]["hidden_dims"] = hidden_dims
        if n_hidden_layers is not None:
            self.config["agent"]["dqn"]["n_hidden_layers"] = n_hidden_layers
        if target_update_freq is not None:
            self.config["agent"]["dqn"]["target_update_freq"] = target_update_freq
        if dropout is not None:
            self.config["agent"]["dqn"]["dropout"] = dropout

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

        self.n_states = 2 * (self.env.target_sum + 1)

        self.tmp_model = None
        self.best_model = None
        self.best_avg_return = -np.inf

    def _update_target_network(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def _greedy_policy(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            action_idx = self.target_network(state).argmax().item()

        return action_idx

    def _select_action(self, state: float, is_eval: bool = False) -> int:
        if is_eval or self.action_rng.random() > self.epsilon:
            if "QTable" in self.name:
                # Denormalize state to use as index
                state = int(state * self.env.target_sum)
                action_idx = argmax(self.Q_sa[state, :], self.action_rng)
            else:
                state = torch.as_tensor(
                    [state], dtype=torch.float32, device=self.device
                )
                action_idx = self._greedy_policy(state)
        else:
            action_idx = int(self.action_rng.integers(self.n_actions))

        return action_idx

    def _update(
        self,
        state: int,
        action_idx: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
        prev_state: int = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        state = int(state * self.env.target_sum)
        next_state = int(next_state * self.env.target_sum)

        current_q_value = self.Q_sa[state, action_idx]

        max_next_q_value = np.max(self.Q_sa[next_state, :]) if not terminated else 0
        next_q_value = reward + self.gamma * (1 - terminated) * max_next_q_value

        self.Q_sa[state, action_idx] += self.alpha * (next_q_value - current_q_value)

    def _add_to_buffer(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        terminated: bool,
        truncated: bool = None,
        prev_state: float = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        self.buffer.add(state, action_idx, reward, next_state, terminated)

    def _learn(self) -> None:
        if self.buffer.real_size < self.batch_size:
            return

        states, action_idxs, rewards, next_states, terminations = self.buffer.sample()

        current_q_values = self.network(states).gather(1, action_idxs)

        with torch.no_grad():
            next_actions = self.network(next_states).argmax(dim=1).view(-1, 1)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + self.gamma * ~terminations * next_q_values

        self.network.optimizer.zero_grad()
        loss = self.network.loss_fn(current_q_values, target_q_values)
        loss.backward()
        self.network.optimizer.step()

    def _train(self) -> None:
        state, terminated, truncated = self.env.reset()

        prev_state = None
        prev_action_idx = None
        prev_reward = None

        train_step = 0
        while train_step < self.n_training_steps:
            action_idx = self._select_action(state)
            next_state, reward, terminated, truncated = self.env.step(state, action_idx)

            if "QTable" in self.name:
                self._update(
                    state,
                    action_idx,
                    reward,
                    next_state,
                    terminated,
                    truncated,
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
                    terminated,
                    truncated,
                    prev_state,
                    prev_action_idx,
                    prev_reward,
                )

            if terminated or truncated:
                if "DQN" in self.name:
                    self._learn()

                prev_state = None
                prev_action_idx = None
                prev_reward = None

                state, terminated, truncated = self.env.reset()
            else:
                prev_state = state
                prev_action_idx = action_idx
                prev_reward = reward

                state = next_state

            train_step += 1

            if "DQN" in self.name:
                if train_step % self.target_update_freq == 0:
                    self._update_target_network()

    def _test(self) -> tuple:
        returns = []
        best_actions = []
        max_return = -np.inf

        for _ in range(self.n_episodes_testing):
            actions = []
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

                actions += [action_idx]

            if episode_reward > max_return:
                max_return = episode_reward
                best_actions = actions

            returns.append(episode_reward)

        return returns, best_actions

    def _plot_best_model(self, returns: float, best_actions: list) -> float:
        avg_returns = np.mean(returns)

        # ALWAYS retrieve average return from best model to plot
        if avg_returns > self.best_avg_return:
            # Average return is better than best so plot/save current model
            self.best_avg_return = avg_returns
            if "QTable" in self.name:
                self.best_model = copy.deepcopy(self.Q_sa)
            else:
                self.best_model = copy.deepcopy(self.target_network.state_dict())
        else:
            # Average return is worse than best so store current model and plot best model
            if "QTable" in self.name:
                tmp_model = copy.deepcopy(self.Q_sa)
                self.Q_sa = copy.deepcopy(self.best_model)
            else:
                tmp_model = copy.deepcopy(self.target_network.state_dict())
                self.target_network.load_state_dict(self.best_model)

            returns, best_actions = self._test()
            avg_returns = np.mean(returns)

            # Restore current model
            if "QTable" in self.name:
                self.Q_sa = copy.deepcopy(tmp_model)
            else:
                self.target_network.load_state_dict(tmp_model)

        return avg_returns, best_actions

    def generate_target_sum(self, problem_instance: str) -> None:
        self._setup_problem(problem_instance)

        current_n_steps = 0
        for _ in range(self.n_episodes):
            returns, best_actions = self._test()

            best_avg_returns, best_actions = self._plot_best_model(
                returns, best_actions
            )
            best_actions = [self.env.elements[action] for action in best_actions]

            log_step = (
                current_n_steps // 3 if "TripleData" in self.name else current_n_steps
            )

            wandb.log(
                {"Average Return": best_avg_returns, "Best Actions": best_actions},
                step=log_step,
            )

            self._train()

            current_n_steps += self.n_training_steps

        wandb.finish()
