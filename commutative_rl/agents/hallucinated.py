from .utils.parent import Parent


class Hallucinated(Parent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        sum_range: range,
        elem_range: range,
        n_elems: int,
        max_noise: float,
    ) -> None:

        super(Hallucinated, self).__init__(
            seed,
            num_instances,
            sum_range,
            elem_range,
            n_elems,
            max_noise,
        )

    def _update(
        self,
        state: int,
        action_idx: int,
        reward: float,
        next_state: int,
        done: bool,
        prev_state: int,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:

        super()._update(state, action_idx, reward, next_state, done)
        super()._update(state, action_idx, reward, next_state, done)
        super()._update(state, action_idx, reward, next_state, done)
