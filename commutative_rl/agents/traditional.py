import numpy as np

from .utils.agent import Agent


class Traditional(Agent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
    ) -> None:

        super(Traditional, self).__init__(seed, num_instances, noise_type)


class TripleTraditional(Traditional):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
    ) -> None:

        super(TripleTraditional, self).__init__(seed, num_instances, noise_type)

        self.n_timesteps *= 3

        self.config["q_table"]["n_timesteps"] = self.n_timesteps
