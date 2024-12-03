import numpy as np

from .utils.agent import Agent


class Traditional(Agent):
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

        super(Traditional, self).__init__(
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

        self.buffer.add(state, action_idx, reward, next_state, done)
        self.buffer.increase_size()


class TripleTraditional(Traditional):
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

        super(TripleTraditional, self).__init__(
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

        self.counter = 0

        self.n_timesteps *= 3
        self.target_update_freq *= 3

        self.config["env"]["n_steps"] = self.n_timesteps
        self.config["agent"]["target_update_freq"] = self.target_update_freq

    def _learn(self, current_n_step: int, step: int) -> None:
        if self.counter % 3 == 0:
            super()._learn(current_n_step, step)

        self.counter += 1
