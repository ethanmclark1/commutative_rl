from .utils.parent import Parent


class Commutative(Parent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        sum_range: range,
        elem_range: range,
        n_elems: int,
        max_noise: float,
    ) -> None:

        super(Commutative, self).__init__(
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

        if prev_state is None or action_idx == 0:
            return

        trace_reward = prev_reward + reward

        commutative_state = prev_state + self.env.elements[action_idx]
        commutative_next_state = commutative_state + self.env.elements[prev_action_idx]

        transition_1 = (prev_state, action_idx, 0, commutative_state, False)
        transition_2 = (
            commutative_state,
            prev_action_idx,
            trace_reward,
            commutative_next_state,
            done,
        )

        for transition in [transition_1, transition_2]:
            super()._update(*transition)
