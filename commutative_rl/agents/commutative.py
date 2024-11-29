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
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        target_update_freq: int = None,
        grad_clip_norm: float = None,
        loss_fn: str = None,
        layer_norm: bool = None,
        aggregation_type: str = None,
    ) -> None:

        super(Commutative, self).__init__(
            seed,
            num_instances,
            sum_range,
            elem_range,
            n_elems,
            max_noise,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            target_update_freq,
            grad_clip_norm,
            loss_fn,
            layer_norm,
            aggregation_type,
        )

    def _get_commutative_state(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        new_elem = self.env.elements[action_idx]

        current_sum = int(state[0] * self.env.target_sum)
        current_n_step = int(state[1] * self.env.n_steps)

        new_sum = (current_sum + new_elem) / self.env.target_sum
        new_n_step = (current_n_step + 1) / self.env.n_steps

        next_state = np.array([new_sum, new_n_step], dtype=float)

        return next_state

    def _add_to_buffer(
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

        self.buffer.add(state, action_idx, reward, next_state, done, 0)

        if prev_state is not None and action_idx != 0:
            if self.aggregation_type == "equal":
                avg_reward = (reward + prev_reward) / 2
                intermediate_reward = avg_reward
                trace_reward = avg_reward
            elif self.aggregation_type == "trace_front":
                intermediate_reward = prev_reward + reward
                trace_reward = 0
            elif self.aggregation_type == "trace_back":
                intermediate_reward = 0
                trace_reward = prev_reward + reward
            elif self.aggregation_type == "mirror_front":
                intermediate_reward = prev_reward
                trace_reward = reward
            elif self.aggregation_type == "mirror_back":
                intermediate_reward = reward
                trace_reward = prev_reward

            commutative_state = self._get_commutative_state(prev_state, action_idx)
            commutative_next_state = self._get_commutative_state(
                commutative_state, prev_action_idx
            )

            transition_1 = (
                prev_state,
                action_idx,
                intermediate_reward,
                commutative_state,
                False,
                1,
            )
            transition_2 = (
                commutative_state,
                prev_action_idx,
                trace_reward,
                commutative_next_state,
                done,
                2,
            )

            for transition in [transition_1, transition_2]:
                self.buffer.add(*transition)

        self.buffer.increase_size()


class CommutativeIndependentSamples(Commutative):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        sum_range: range,
        elem_range: range,
        n_elems: int,
        max_noise: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        target_update_freq: int = None,
        grad_clip_norm: float = None,
        loss_fn: str = None,
        layer_norm: bool = None,
        aggregation_type: str = None,
    ) -> None:

        super(CommutativeIndependentSamples, self).__init__(
            seed,
            num_instances,
            sum_range,
            elem_range,
            n_elems,
            max_noise,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            target_update_freq,
            grad_clip_norm,
            loss_fn,
            layer_norm,
            aggregation_type,
        )

    def _add_to_buffer(
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

        self.buffer.add(state, action_idx, reward, next_state, done, 0)
        self.buffer.increase_size()

        if prev_state is not None and action_idx != 0:
            if self.aggregation_type == "equal":
                avg_reward = (reward + prev_reward) / 2
                intermediate_reward = avg_reward
                trace_reward = avg_reward
            elif self.aggregation_type == "trace_front":
                intermediate_reward = prev_reward + reward
                trace_reward = 0
            elif self.aggregation_type == "trace_back":
                intermediate_reward = 0
                trace_reward = prev_reward + reward
            elif self.aggregation_type == "mirror_front":
                intermediate_reward = prev_reward
                trace_reward = reward
            elif self.aggregation_type == "mirror_back":
                intermediate_reward = reward
                trace_reward = prev_reward

            commutative_state = self._get_commutative_state(prev_state, action_idx)
            commutative_next_state = self._get_commutative_state(
                commutative_state, prev_action_idx
            )

            transition_1 = (
                prev_state,
                action_idx,
                intermediate_reward,
                commutative_state,
                False,
                0,
            )
            transition_2 = (
                commutative_state,
                prev_action_idx,
                trace_reward,
                commutative_next_state,
                done,
                0,
            )

            for transition in [transition_1, transition_2]:
                self.buffer.add(*transition)
                self.buffer.increase_size()
