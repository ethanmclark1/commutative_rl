from .utils.agent import Agent


"""
Exact Methods
------------------------------------------------------------------------------------------------------------------------
"""


class TraditionalQTable(Agent):
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

        super(TraditionalQTable, self).__init__(
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


class TripleTraditionalQTable(TraditionalQTable):
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

        super(TripleTraditionalQTable, self).__init__(
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

        self.n_training_steps *= 3

        self.config["agent"]["n_training_steps"] = self.n_training_steps


"""
Approximate Methods
------------------------------------------------------------------------------------------------------------------------
"""


class TraditionalDQN(Agent):
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

        super(TraditionalDQN, self).__init__(
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


class TripleTraditionalDQN(TraditionalDQN):
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

        super(TripleTraditionalDQN, self).__init__(
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

        self.n_training_steps *= 3

        self.config["agent"]["n_training_steps"] = self.n_training_steps
