import copy
import numpy as np

from .utils.agent import Agent
from .utils.networks import MLP
from .utils.buffers import ReplayBuffer


"""
Exact Approaches
----------------
"""


class QTable(Agent):
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

        super(QTable, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
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

    def _setup_problem(self, problem_instance) -> None:
        Agent._setup_problem(self, problem_instance)
        self.Q_sa = np.zeros((self.n_states, self.n_actions))


class TripleDataQTable(QTable):
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

        super(TripleDataQTable, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
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

        self.n_training_steps *= 3

        self.config["agent"]["n_training_steps"] = self.n_training_steps


"""
Approximate Approaches
----------------------
"""


class DQN(Agent):
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

        super(DQN, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
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

    def _setup_problem(self, problem_instance) -> None:
        Agent._setup_problem(self, problem_instance)

        self.network = MLP(
            self.seed,
            self.state_dims,
            self.n_actions,
            self.hidden_dims,
            self.n_hidden_layers,
            self.alpha,
            self.dropout,
        ).to(self.device)

        self.target_network = copy.deepcopy(self.network)
        self.buffer = ReplayBuffer(
            self.seed,
            self.state_dims,
            self.batch_size,
            self.buffer_size,
            self.device,
        )

        self.network.train()
        self.target_network.eval()


class TripleDataDQN(DQN):
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

        super(TripleDataDQN, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
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

        self.n_training_steps *= 3

        self.config["agent"]["n_training_steps"] = self.n_training_steps
