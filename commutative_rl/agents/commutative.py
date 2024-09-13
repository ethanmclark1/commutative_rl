import numpy as np

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
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        episode_step: int,
        prev_state: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:

        super()._update(
            state, action_idx, reward, next_state, terminated, truncated, episode_step
        )

        if prev_state is None or action_idx == 0:
            return

        trace_reward = prev_reward + reward

        commutative_state, next_state = self._reassign_states(
            prev_state, prev_action_idx, state, action_idx, next_state
        )

        transition_1 = (
            prev_state,
            action_idx,
            0,
            commutative_state,
            False,
            False,
            episode_step - 1,
        )
        transition_2 = (
            commutative_state,
            prev_action_idx,
            trace_reward,
            next_state,
            terminated,
            truncated,
            episode_step,
        )

        for transition in [transition_1, transition_2]:
            super()._update(*transition)
