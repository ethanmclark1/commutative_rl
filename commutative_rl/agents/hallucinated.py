import numpy as np

from .utils.parent import Parent


class Hallucinated(Parent):
    def __init__(self, seed, num_instances, noise_type) -> None:

        super(Hallucinated, self).__init__(seed, num_instances, noise_type)

    def _update(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
        episode_step: int,
        prev_state: np.ndarray = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        super()._update(
            state, action_idx, reward, next_state, terminated, truncated, episode_step
        )
        super()._update(
            state, action_idx, reward, next_state, terminated, truncated, episode_step
        )
        super()._update(
            state, action_idx, reward, next_state, terminated, truncated, episode_step
        )
