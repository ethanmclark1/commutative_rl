from .utils.parent import Parent


class Traditional(Parent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: float,
    ) -> None:

        super(Traditional, self).__init__(seed, num_instances, noise_type)
