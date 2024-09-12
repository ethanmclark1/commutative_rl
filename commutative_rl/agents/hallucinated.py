from .utils.parent import Parent


class Hallucinated(Parent):
    def __init__(self, seed, num_instances, noise_type) -> None:

        super(Hallucinated, self).__init__(seed, num_instances, noise_type)

    def _update(
        self,
        state: int,
        action_idx: int,
        reward: float,
        next_state: int,
        done: bool,
        prev_state: int = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        super()._update(state, action_idx, reward, next_state, done)
        super()._update(state, action_idx, reward, next_state, done)
        super()._update(state, action_idx, reward, next_state, done)
