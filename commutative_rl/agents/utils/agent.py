import os
import yaml
import copy
import torch
import wandb
import numpy as np

from env import Env

from .helpers import *


class Agent:
    def __init__(
        self,
        seed: int,
        n_agents: int,
        n_large_obstacles: int,
        n_small_obstacles: int,
        n_episode_steps: int,
        granularity: float,
        safe_area_multiplier: float,
        failed_path_cost: float,
        configs_to_consider: int,
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
            n_agents,
            n_large_obstacles,
            n_small_obstacles,
            n_episode_steps,
            granularity,
            safe_area_multiplier,
            failed_path_cost,
            configs_to_consider,
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
            n_agents,
            n_large_obstacles,
            n_small_obstacles,
            self.config["env"],
        )

        self.state_dims = self.env.state_dims
        self.n_actions = self.env.n_actions

        self.action_rng = np.random.default_rng(seed)

        self.tmp_model = None
        self.best_model = None
        self.best_avg_return = -np.inf

    def _init_params(
        self,
        filepath: str,
        seed: int,
        n_agents: int,
        n_large_obstacles: int,
        n_small_obstacles: int,
        n_episode_steps: int,
        granularity: float,
        safe_area_multiplier: float,
        failed_path_cost: float,
        configs_to_consider: int,
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

        # Override default env values with command line arguments
        if n_agents is not None:
            self.config["env"]["n_agents"] = n_agents
        if n_large_obstacles is not None:
            self.config["env"]["n_large_obstacles"] = n_large_obstacles
        if n_small_obstacles is not None:
            self.config["env"]["n_small_obstacles"] = n_small_obstacles
        if n_episode_steps is not None:
            self.config["env"]["n_episode_steps"] = n_episode_steps
        if granularity is not None:
            self.config["env"]["granularity"] = granularity
        if safe_area_multiplier is not None:
            self.config["env"]["safe_area_multiplier"] = safe_area_multiplier
        if failed_path_cost is not None:
            self.config["env"]["failed_path_cost"] = failed_path_cost
        if configs_to_consider is not None:
            self.config["env"]["configs_to_consider"] = configs_to_consider

        # Override default agent values with command line arguments
        if alpha is not None:
            self.config["agent"]["dqn"]["alpha"] = alpha
        if epsilon is not None:
            self.config["agent"]["dqn"]["epsilon"] = epsilon
        if gamma is not None:
            self.config["agent"]["dqn"]["gamma"] = gamma
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
            project="Language Domain",
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

        wandb.config["terminal_reward"] = self.env.terminal_reward

    def _setup_problem(self, problem_instance: str) -> None:
        self.env.set_problem(problem_instance)
        self._setup_wandb(problem_instance)

    def _update_target_network(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def _select_action(self, state: list, is_eval: bool = False) -> int:
        if is_eval or self.action_rng.random() > self.epsilon:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action_idx = self._greedy_policy(state)
        else:
            action_idx = self.action_rng.integers(self.n_actions)

        return action_idx

    def _add_to_buffer(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        terminated: bool,
        truncated: bool,
        prev_state: float = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:
        raise NotImplementedError

    def _learn(self) -> None:
        raise NotImplementedError

    def _train(self) -> None:
        state, terminated, truncated = self.env.reset()

        prev_state = None
        prev_action_idx = None
        prev_reward = None

        train_step = 0
        while train_step < self.n_training_steps:
            action_idx = self._select_action(state)
            next_state, reward, terminated, truncated = self.env.step(state, action_idx)

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

            if train_step % self.target_update_freq == 0:
                self._update_target_network()

    def _test(self) -> float:
        returns = []

        for _ in range(self.n_episodes_testing):
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

    def _plot_best_model(self, returns: float) -> float:
        avg_returns = np.mean(returns)

        # ALWAYS retrieve average return from best model to plot
        if avg_returns > self.best_avg_return:
            # Average return is better than best so plot/save current model
            self.best_avg_return = avg_returns
            self.best_model = copy.deepcopy(self.target_network.state_dict())
        else:
            # Average return is worse than best so store current model and plot best model
            tmp_model = copy.deepcopy(self.target_network.state_dict())
            self.target_network.load_state_dict(self.best_model)

            returns = self._test()
            avg_returns = np.mean(returns)

            # Restore current model
            self.target_network.load_state_dict(tmp_model)

        return avg_returns

    def generate_language(self, problem_instance: str) -> None:
        self._setup_problem(problem_instance)

        current_n_steps = 0
        for _ in range(self.n_episodes):
            self._train()
            returns = self._test()

            avg_returns = self._plot_best_model(returns)

            log_step = (
                current_n_steps // 3
                if "TripleTraditional" in self.name
                else current_n_steps
            )

            wandb.log({"Average Return": avg_returns}, step=log_step)

            current_n_steps += self.n_training_steps

        wandb.finish()
