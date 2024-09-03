import torch
import numpy as np

from .utils.parent import Parent
from .utils.buffers import ReplayBuffer, RewardBuffer


class Commutative(Parent):
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

        super(Commutative, self).__init__(
            seed,
            num_instances,
            min_dist_bounds,
            action_dims,
            negative_actions,
            duplicate_actions,
            reward_type,
            reward_noise,
        )

        self.commutative_reward_buffer = None
        self.commutative_replay_buffer = None

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

        action = self.actions[action_idx]
        if prev_state is not None:
            if self.reward_type == "true":
                prev_commutative_reward, commutative_state, _ = self._step(
                    prev_state, action_idx, num_action - 1
                )

                if action == 0:
                    prev_action_idx = action_idx
                    commutative_reward = 0
                    commutative_next_state = commutative_state
                else:
                    commutative_reward, commutative_next_state, done = self._step(
                        commutative_state, prev_action_idx, num_action
                    )

                self.commutative_replay_buffer.add(
                    prev_state,
                    action_idx,
                    prev_commutative_reward,
                    commutative_state,
                    prev_action_idx,
                    commutative_reward,
                    commutative_next_state,
                    done,
                    num_action,
                )

            else:
                commutative_state = self._get_next_state(prev_state, action)

                if action == 0:
                    prev_action_idx = action_idx
                    prev_reward = reward
                    commutative_next_state = commutative_state
                else:
                    prev_action = self.actions[prev_action_idx]
                    commutative_next_state = self._get_next_state(
                        commutative_state, prev_action
                    )

                self.commutative_replay_buffer.add(
                    prev_state,
                    action_idx,
                    -1,
                    commutative_state,
                    prev_action_idx,
                    -1,
                    commutative_next_state,
                    done,
                    num_action,
                )

                self.commutative_reward_buffer.add(
                    prev_state,
                    action_idx,
                    prev_reward,
                    commutative_state,
                    prev_action_idx,
                    reward,
                    commutative_next_state,
                    num_action,
                )

            self.commutative_traces += 1

    def _update_estimator(
        self,
        losses: dict,
        reward_buffer: object = None,
        loss_type: str = None,
        indices: tuple = None,
    ) -> None:
        self.estimator.optim.zero_grad()

        indices = super()._update_estimator(losses, self.reward_buffer, "step_loss")

        if self.estimator_batch_size > self.commutative_reward_buffer.real_size:
            return

        transitions = self.commutative_reward_buffer.transition[indices]
        rewards = self.commutative_reward_buffer.reward[indices]

        summed_rewards = torch.sum(rewards, dim=1, keepdim=True)
        r2_pred = self.estimator(transitions[:, 0])
        r3_pred = self.estimator(transitions[:, 1])

        loss_r2 = self.estimator.loss(r2_pred + r3_pred.detach(), summed_rewards)
        loss_r3 = self.estimator.loss(r2_pred.detach() + r3_pred, summed_rewards)
        trace_loss = loss_r2 + loss_r3
        trace_loss.backward()

        losses["trace_loss"] += abs(trace_loss.item() / 2)

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
            losses, self.commutative_replay_buffer, "commutative_loss", indices
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

        self.commutative_replay_buffer = ReplayBuffer(
            self.seed, self.max_elements, 1, self.buffer_size, self.action_dims
        )
        self.commutative_reward_buffer = RewardBuffer(
            self.seed,
            self.step_dims,
            self.max_elements,
            self.estimator_buffer_size,
            self.action_dims,
        )

        super().generate_target_sum(problem_instance)
