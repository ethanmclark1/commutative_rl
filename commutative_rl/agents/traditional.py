import copy
import torch
import numpy as np

from .utils.agent import Agent
from .utils.networks import DQN
from .utils.buffers import ReplayBuffer


class TraditionalDQN(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        grid_dims: str,
        n_starts: int,
        n_goals: int,
        n_bridges: int,
        n_episode_steps: int,
        action_success_rate: float,
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

        super(TraditionalDQN, self).__init__(
            seed,
            n_instances,
            grid_dims,
            n_starts,
            n_goals,
            n_bridges,
            n_episode_steps,
            action_success_rate,
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

        self.network = DQN(
            seed,
            self.state_dims,
            self.n_actions,
            self.hidden_dims,
            self.n_hidden_layers,
            self.alpha,
            self.dropout,
        ).to(self.device)

        self.target_network = copy.deepcopy(self.network)
        self.buffer = ReplayBuffer(
            seed, self.state_dims, self.batch_size, self.buffer_size, self.device
        )

        self.network.train()
        self.target_network.eval()

    def _greedy_policy(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            action_idx = self.target_network(state).argmax().item()

        return action_idx

    def _add_to_buffer(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        prev_state: float = np.ndarray,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        self.buffer.add(state, action_idx, reward, next_state, terminated)
        self._learn()

    def _learn(self) -> None:
        if self.buffer.real_size < self.batch_size:
            return

        states, action_idxs, rewards, next_states, terminations = self.buffer.sample()

        for i in range(self.batch_size):
            with torch.no_grad():
                next_q_values = self.target_network(next_states[i])
                max_next_q_value = torch.max(next_q_values)
                target = rewards[i] + self.gamma * ~terminations[i] * max_next_q_value

                target_q_values = self.target_network(states[i])
                target_q_values[0, action_idxs[i]] = target

            current_q_values = self.network(states[i])

            self.network.optimizer.zero_grad()
            loss = self.network.loss_fn(current_q_values, target_q_values)
            loss.backward()
            self.network.optimizer.step()


class TripleTraditionalDQN(TraditionalDQN):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        grid_dims: str,
        n_starts: int,
        n_goals: int,
        n_bridges: int,
        n_episode_steps: int,
        action_success_rate: float,
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

        super(TripleTraditionalDQN, self).__init__(
            seed,
            n_instances,
            grid_dims,
            n_starts,
            n_goals,
            n_bridges,
            n_episode_steps,
            action_success_rate,
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

        self.n_training_steps *= 3

        self.config["agent"]["n_training_steps"] = self.n_training_steps
