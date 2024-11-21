import numpy as np

from .utils.agent import Agent


class Commutative(Agent):
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

    def _get_commutative_state(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        new_elem = self.env.elements[action_idx]

        current_sum = int(state[0] * self.env.target_sum)
        current_n_step = int(state[1] * self.env.n_steps)

        new_sum = (current_sum + new_elem) / self.env.target_sum
        new_n_step = (current_n_step + 1) / self.env.n_steps

        next_state = np.array([new_sum, new_n_step], dtype=float)

        return next_state

    def _update(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        prev_state: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:

        super()._update(state, action_idx, reward, next_state, done)

        if prev_state is not None and action_idx != 0:
            trace_reward = prev_reward + reward

            commutative_state = self._get_commutative_state(prev_state, action_idx)
            commutative_next_state = self._get_commutative_state(
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
