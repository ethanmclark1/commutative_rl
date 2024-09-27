import numpy as np

from .utils.agent import Agent


class TripleData(Agent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
    ) -> None:

        super(TripleData, self).__init__(seed, num_instances, noise_type)
        self.n_timesteps *= 3

        self.counter = 0

    def _add_to_buffers(
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

        super()._add_to_buffers(
            state,
            action_idx,
            reward,
            next_state,
            done,
            episode_step,
        )

        self.replay_buffer.increase_size()

    def _learn(self):
        if self.counter % 3 == 0:
            super()._learn()

        self.counter += 1
