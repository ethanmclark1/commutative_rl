import numpy as np

from .utils.agent import Agent


# generate a commutative state without using the environment noise
def get_commutative_state(
    env: object, state: np.ndarray, action_idx: int
) -> np.ndarray:
    new_elem = env.elements[action_idx]

    current_sum = int(state[0] * env.target_sum)
    current_n_step = int(state[1] * env.n_steps)

    new_sum = (current_sum + new_elem) / env.target_sum
    new_n_step = (current_n_step + 1) / env.n_steps

    next_state = np.array([new_sum, new_n_step], dtype=float)

    return next_state


"""
Exact Methods
------------------------------------------------------------------------------------------------------------------------
"""


class CommutativeQTable(Agent):
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

        super(CommutativeQTable, self).__init__(
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
        episode_step: int,
    ) -> None:

        super()._update(state, action_idx, reward, next_state, done)

        if prev_state is not None and action_idx != 0:
            commutative_state = get_commutative_state(self.env, prev_state, action_idx)
            commutative_next_state = get_commutative_state(
                self.env, commutative_state, prev_action_idx
            )

            if self.aggregation_type == "equal":
                avg_reward = (reward + prev_reward) / 2
                intermediate_reward = avg_reward
                trace_reward = avg_reward
                prev_done = False
            elif self.aggregation_type == "trace_front":
                intermediate_reward = prev_reward + reward
                trace_reward = 0
                prev_done = False
            elif self.aggregation_type == "trace_back":
                intermediate_reward = 0
                trace_reward = prev_reward + reward
                prev_done = False
            elif self.aggregation_type == "mirror_front":
                intermediate_reward = prev_reward
                trace_reward = reward
                prev_done = False
            elif self.aggregation_type == "mirror_back":
                intermediate_reward = reward
                trace_reward = prev_reward
                prev_done = False
            elif self.aggregation_type == "true_reward":
                commutative_state, intermediate_reward, prev_done = self.env.step(
                    prev_state, action_idx, episode_step - 1
                )
                commutative_next_state, trace_reward, done = self.env.step(
                    commutative_state, prev_action_idx, episode_step
                )

            transition_1 = (
                prev_state,
                action_idx,
                intermediate_reward,
                commutative_state,
                prev_done,
            )
            transition_2 = (
                commutative_state,
                prev_action_idx,
                trace_reward,
                commutative_next_state,
                done,
            )

            for transition in [transition_1, transition_2]:
                super()._update(*transition)


"""
Function Approximation Methods
------------------------------------------------------------------------------------------------------------------------
"""


class CommutativeDQN(Agent):
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

        super(CommutativeDQN, self).__init__(
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
        episode_step: int,
    ) -> None:

        self.buffer.add(state, action_idx, reward, next_state, done, 0)

        if prev_state is not None and action_idx != 0:
            commutative_state = get_commutative_state(self.env, prev_state, action_idx)
            commutative_next_state = get_commutative_state(
                self.env, commutative_state, prev_action_idx
            )

            if self.aggregation_type == "equal":
                avg_reward = (reward + prev_reward) / 2
                intermediate_reward = avg_reward
                trace_reward = avg_reward
                prev_done = False
            elif self.aggregation_type == "trace_front":
                intermediate_reward = prev_reward + reward
                trace_reward = 0
                prev_done = False
            elif self.aggregation_type == "trace_back":
                intermediate_reward = 0
                trace_reward = prev_reward + reward
                prev_done = False
            elif self.aggregation_type == "mirror_front":
                intermediate_reward = prev_reward
                trace_reward = reward
                prev_done = False
            elif self.aggregation_type == "mirror_back":
                intermediate_reward = reward
                trace_reward = prev_reward
                prev_done = False
            elif self.aggregation_type == "true_reward":
                commutative_state, intermediate_reward, prev_done = self.env.step(
                    prev_state, action_idx, episode_step - 1
                )
                commutative_next_state, trace_reward, done = self.env.step(
                    commutative_state, prev_action_idx, episode_step
                )

            transition_1 = (
                prev_state,
                action_idx,
                intermediate_reward,
                commutative_state,
                prev_done,
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


class CommutativeIndependentSamplesDQN(CommutativeDQN):
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

        super(CommutativeIndependentSamplesDQN, self).__init__(
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
        episode_step: int,
    ) -> None:

        self.buffer.add(state, action_idx, reward, next_state, done, 0)
        self.buffer.increase_size()

        if prev_state is not None and action_idx != 0:
            commutative_state = get_commutative_state(self.env, prev_state, action_idx)
            commutative_next_state = get_commutative_state(
                self.env, commutative_state, prev_action_idx
            )

            if self.aggregation_type == "equal":
                avg_reward = (reward + prev_reward) / 2
                intermediate_reward = avg_reward
                trace_reward = avg_reward
                prev_done = False
            elif self.aggregation_type == "trace_front":
                intermediate_reward = prev_reward + reward
                trace_reward = 0
                prev_done = False
            elif self.aggregation_type == "trace_back":
                intermediate_reward = 0
                trace_reward = prev_reward + reward
                prev_done = False
            elif self.aggregation_type == "mirror_front":
                intermediate_reward = prev_reward
                trace_reward = reward
                prev_done = False
            elif self.aggregation_type == "mirror_back":
                intermediate_reward = reward
                trace_reward = prev_reward
                prev_done = False
            elif self.aggregation_type == "true_reward":
                commutative_state, intermediate_reward, prev_done = self.env.step(
                    prev_state, action_idx, episode_step - 1
                )
                commutative_next_state, trace_reward, done = self.env.step(
                    commutative_state, prev_action_idx, episode_step
                )

            transition_1 = (
                prev_state,
                action_idx,
                intermediate_reward,
                commutative_state,
                prev_done,
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
