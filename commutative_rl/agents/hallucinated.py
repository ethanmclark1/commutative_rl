import numpy as np

from .utils.parent import Parent


class Hallucinated(Parent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
    ) -> None:

        super(Hallucinated, self).__init__(seed, num_instances, noise_type)

    def _add_to_buffers(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        episode_step: int,
        prev_state: np.ndarray = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        super()._add_to_buffers(
            self.replay_buffer,
            state,
            action_idx,
            reward,
            next_state,
            done,
            episode_step,
        )

    def _learn(self, losses) -> None:
        indices = super()._learn(
            losses, self.replay_buffer, loss_type="traditional_loss"
        )

        if indices is None:
            return

        super()._learn(losses, self.replay_buffer, indices, "hallucinated_loss")
