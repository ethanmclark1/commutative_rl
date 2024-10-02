import numpy as np

from .utils.agent import Agent


class TripleData(Agent):
    def __init__(
        self,
        seed: int,
        num_agents: int,
        num_large_obstacles: int,
        num_small_obstacles: int,
        config: dict,
    ) -> None:

        super(TripleData, self).__init__(
            seed, num_agents, num_large_obstacles, num_small_obstacles, config
        )

    def _add_to_buffers(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        episode_step: int,
        prev_state: np.ndarray,
        prev_action_idx,
        prev_reward,
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
