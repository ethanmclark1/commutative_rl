from .utils.agent import Agent


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
        learning_start_step: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        activation_fn: str = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        grad_clip_norm: float = None,
        loss_fn: str = None,
        layer_norm: bool = None,
        step_scale: float = None,
        over_penalty: float = None,
        under_penalty: float = None,
        complete_reward: float = None,
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
            learning_start_step,
            buffer_size,
            hidden_dims,
            activation_fn,
            n_hidden_layers,
            target_update_freq,
            grad_clip_norm,
            loss_fn,
            layer_norm,
            step_scale,
            over_penalty,
            under_penalty,
            complete_reward,
        )

    def _update(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        done: bool,
        prev_state: float,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:

        super()._update(state, action_idx, reward, next_state, done)

        if prev_state is None or action_idx == 0:
            return

        action_a = state - prev_state
        action_b = next_state - state

        commutative_state = prev_state + action_b
        commutative_next_state = commutative_state + action_a

        trace_reward = prev_reward + reward

        transition_1 = (
            prev_state,
            action_idx,
            0,
            commutative_state,
            False,
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
Approximate Methods
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
        learning_start_step: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        activation_fn: str = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        grad_clip_norm: float = None,
        loss_fn: str = None,
        layer_norm: bool = None,
        step_scale: float = None,
        over_penalty: float = None,
        under_penalty: float = None,
        complete_reward: float = None,
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
            learning_start_step,
            buffer_size,
            hidden_dims,
            activation_fn,
            n_hidden_layers,
            target_update_freq,
            grad_clip_norm,
            loss_fn,
            layer_norm,
            step_scale,
            over_penalty,
            under_penalty,
            complete_reward,
        )

    def _add_to_buffer(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        done: bool,
        prev_state: float,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:

        super()._add_to_buffer(state, action_idx, reward, next_state, done)

        if prev_state is None or action_idx == 0:
            return

        noisy_action_a = state - prev_state
        noisy_action_b = next_state - state

        commutative_state = prev_state + noisy_action_b
        commutative_next_state = commutative_state + noisy_action_a

        trace_reward = prev_reward + reward

        transition_1 = (
            prev_state,
            action_idx,
            0,
            commutative_state,
            False,
        )
        transition_2 = (
            commutative_state,
            prev_action_idx,
            trace_reward,
            commutative_next_state,
            done,
        )

        for transition in [transition_1, transition_2]:
            super()._add_to_buffer(*transition)
