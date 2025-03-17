import numpy as np

from .utils.agent import Agent


class QTable(Agent):
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
        utility_scale: float,
        terminal_reward: int,
        early_termination_penalty: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
    ) -> None:

        super(QTable, self).__init__(
            seed,
            n_instances,
            grid_dims,
            n_starts,
            n_goals,
            n_bridges,
            n_episode_steps,
            action_success_rate,
            utility_scale,
            terminal_reward,
            early_termination_penalty,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
        )

    def _setup_problem(self, problem_instance) -> None:
        Agent._setup_problem(self, problem_instance)
        self.Q_sa = np.zeros((self.n_states, self.n_actions))


class TripleDataQTable(QTable):
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
        utility_scale: float,
        terminal_reward: int,
        early_termination_penalty: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
    ) -> None:

        super(TripleDataQTable, self).__init__(
            seed,
            n_instances,
            grid_dims,
            n_starts,
            n_goals,
            n_bridges,
            n_episode_steps,
            action_success_rate,
            utility_scale,
            terminal_reward,
            early_termination_penalty,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
        )

        self.n_training_steps *= 3

        self.config["agent"]["n_training_steps"] = self.n_training_steps
