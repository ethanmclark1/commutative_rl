import copy
import torch
import numpy as np

from .utils.agent import Agent
from .utils.networks import DQN
from .utils.buffers import ReplayBuffer

"""
Exact Methods
------------------------------------------------------------------------------------------------------------------------
"""


class TraditionalQTable(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
        step_value: float = None,
        over_penalty: float = None,
    ) -> None:

        super(TraditionalQTable, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
            step_value,
            over_penalty,
        )

    def _init_q_table(self, n_states: int) -> None:
        self.Q_sa = np.zeros((n_states * 2, self.n_actions))

    def _update(
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

        state = int(state * self.env.target_sum)
        next_state = int(next_state * self.env.target_sum)

        current_q_value = self.Q_sa[state, action_idx]

        max_next_q_value = np.max(self.Q_sa[next_state, :]) if not terminated else 0
        next_q_value = reward + self.gamma * (1 - terminated) * max_next_q_value

        self.Q_sa[state, action_idx] += self.alpha * (next_q_value - current_q_value)


class TripleTraditionalQTable(TraditionalQTable):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
        step_value: float = None,
        over_penalty: float = None,
    ) -> None:

        super(TripleTraditionalQTable, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
            step_value,
            over_penalty,
        )

        self.n_training_steps *= 3

        self.config["agent"]["n_training_steps"] = self.n_training_steps


"""
Approximate Methods
------------------------------------------------------------------------------------------------------------------------
"""


class TraditionalDQN(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
        step_value: float = None,
        over_penalty: float = None,
    ) -> None:

        super(TraditionalDQN, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
            step_value,
            over_penalty,
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
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
        step_value: float = None,
        over_penalty: float = None,
    ) -> None:

        super(TripleTraditionalDQN, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
            step_value,
            over_penalty,
        )

        self.n_training_steps *= 3

        self.config["agent"]["n_training_steps"] = self.n_training_steps
