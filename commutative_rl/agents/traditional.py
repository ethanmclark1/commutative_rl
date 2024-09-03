import numpy as np

from .utils.parent import Parent
from .utils.buffers import ReplayBuffer, RewardBuffer


class Traditional(Parent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        min_dist_bounds: int,
        action_dims: int,
        negative_actions: bool,
        duplicate_actions: bool,
        reward_type: str,
        reward_noise: float,
    ) -> None:

        super(Traditional, self).__init__(
            seed,
            num_instances,
            min_dist_bounds,
            action_dims,
            negative_actions,
            duplicate_actions,
            reward_type,
            reward_noise,
        )

        self.traditional_reward_buffer = None
        self.traditional_replay_buffer = None

    def _add_to_buffers(
        self,
        prev_state: list,
        prev_action_idx: int,
        prev_reward: float,
        state: list,
        action_idx: int,
        reward: float,
        next_state: list,
        done: bool,
        num_action: int,
    ) -> None:

        super()._add_to_buffers(
            prev_state,
            prev_action_idx,
            prev_reward,
            state,
            action_idx,
            reward,
            next_state,
            done,
            num_action,
        )

        if prev_state is not None:
            self.traditional_replay_buffer.add(
                prev_state,
                prev_action_idx,
                prev_reward,
                state,
                action_idx,
                reward,
                next_state,
                done,
                num_action,
            )

    def _update_estimator(
        self,
        losses: dict,
        reward_buffer: object = None,
        loss_type: str = None,
        indices: tuple = None,
    ) -> None:
        self.estimator.optim.zero_grad()

        indices = super()._update_estimator(losses, self.reward_buffer, "step_loss")
        super()._update_estimator(
            losses, self.traditional_reward_buffer, "step_loss", indices
        )

        self.estimator.optim.step()

    def _learn(
        self,
        losses: dict,
        replay_buffer: object = None,
        loss_type: str = None,
        indices: tuple = None,
    ) -> None:
        self.dqn.optim.zero_grad()

        indices = super()._learn(losses, self.replay_buffer, "traditional_loss")
        super()._learn(
            losses, self.traditional_replay_buffer, "traditional_loss", indices
        )

        self.dqn.optim.step()

        for target_param, local_param in zip(
            self.target_dqn.parameters(), self.dqn.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def generate_target_sum(self, problem_instance: str) -> np.ndarray:
        self._set_problem(problem_instance)

        self.traditional_replay_buffer = ReplayBuffer(
            self.seed, self.max_elements, 1, self.buffer_size, self.action_dims
        )
        self.traditional_reward_buffer = RewardBuffer(
            self.seed,
            self.step_dims,
            self.max_elements,
            self.estimator_buffer_size,
            self.action_dims,
        )

        super().generate_target_sum(problem_instance)
