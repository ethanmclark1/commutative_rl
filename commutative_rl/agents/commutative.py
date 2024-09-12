from .utils.parent import Parent


class Commutative(Parent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: float,
    ) -> None:

        super(Commutative, self).__init__(seed, num_instances, noise_type)

    def _reassign_states(
        self,
        prev_state: np.ndarray,
        prev_action: int,
        state: np.ndarray,
        action_idx: int,
        next_state: np.ndarray,
    ) -> tuple:

        action_a_success = not np.array_equal(prev_state, state)
        action_b_success = not np.array_equal(state, next_state)

        commutative_state = self.env.place_bridge(prev_state, action_idx)

        if action_a_success and action_b_success:
            pass
        elif not action_a_success and action_b_success:
            if prev_action != action_idx:
                next_state = commutative_state
        elif action_a_success and not action_b_success:
            commutative_state = prev_state
            next_state = state
        else:
            commutative_state = prev_state

        return commutative_state, next_state

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

        if prev_state is None or action_idx == 0:
            return

        trace_reward = prev_reward + reward

        commutative_state = self.env.place_bridge(prev_state, action_idx)
        commutative_next_state = self.env.place_bridge(
            commutative_state, prev_action_idx
        )

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
