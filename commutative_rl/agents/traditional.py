from .utils.agent import Agent


class QTable(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        step_value: float = None,
        over_penalty: float = None,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
    ) -> None:

        super(QTable, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            step_value,
            over_penalty,
            alpha,
            epsilon,
            gamma,
        )


class TripleDataQTable(QTable):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        step_value: float = None,
        over_penalty: float = None,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
    ) -> None:

        super(TripleDataQTable, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            step_value,
            over_penalty,
            alpha,
            epsilon,
            gamma,
        )

        self.n_training_steps *= 3

        self.config["agent"]["n_training_steps"] = self.n_training_steps
