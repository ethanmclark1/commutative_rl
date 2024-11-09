import numpy as np

from .utils.agent import Agent


class Traditional(Agent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
        alpha: float,
        buffer_size: int,
        target_update_freq: int,
    ) -> None:

        super(Traditional, self).__init__(
            seed,
            num_instances,
            noise_type,
            alpha,
            buffer_size,
            target_update_freq,
        )

    def _add_to_buffer(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        episode_step: int,
        prev_state: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:

        super()._add_to_buffer(
            state,
            action_idx,
            reward,
            next_state,
            done,
            episode_step,
        )

        self.replay_buffer.increase_size()


class TripleTraditional(Traditional):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
        alpha: float,
        buffer_size: int,
        target_update_freq: int,
    ) -> None:

        super(TripleTraditional, self).__init__(
            seed,
            num_instances,
            noise_type,
            alpha,
            buffer_size,
            target_update_freq,
        )

        self.counter = 0

        self.n_timesteps *= 3
        self.buffer_size *= 3
        self.target_update_freq *= 3

        self.config["dqn"]["n_timesteps"] = self.n_timesteps
        self.config["dqn"]["buffer_size"] = self.buffer_size
        self.config["dqn"]["target_update_freq"] = self.target_update_freq

    def _learn(self) -> None:
        if self.counter % 3 == 0:
            super()._learn()

        self.counter += 1
