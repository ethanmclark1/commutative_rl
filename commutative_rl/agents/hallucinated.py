import numpy as np
import more_itertools

from collections import Counter

from .utils.parent import Parent
from .utils.buffers import ReplayBuffer, RewardBuffer


class Hallucinated(Parent):
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

        super(Hallucinated, self).__init__(
            seed,
            num_instances,
            min_dist_bounds,
            action_dims,
            negative_actions,
            duplicate_actions,
            reward_type,
            reward_noise,
        )

        self.hallucinated_reward_buffer = None
        self.hallucinated_replay_buffer = None

    def _hallucinate(self, non_zero_elements: list, with_approximate: bool) -> None:
        powerset = sorted(list(more_itertools.powerset(non_zero_elements)))

        prev_state = None

        for subset in powerset:
            state = sorted(list(subset)) + [0] * (self.max_elements - len(subset))
            actions = [
                element for element in non_zero_elements if element not in subset
            ]

            if prev_state is not None:
                prev_state_counter = Counter(prev_state)
                state_counter = Counter(state)

                del prev_state_counter[0]
                del state_counter[0]

                difference = state_counter - prev_state_counter

                try:
                    prev_action = list(difference.keys())[0]
                    prev_action_idx = self.actions.index(prev_action)
                    prev_reward, _, _ = self._step(
                        prev_state, prev_action_idx, len(subset) + 1
                    )
                except IndexError:
                    continue

            if len(subset) != self.max_elements:
                actions.append(0)

            num_action = len(subset) + 1
            for action in actions:
                if prev_state is None:
                    break

                action_idx = self.actions.index(action)
                reward, next_state, done = self._step(state, action_idx, num_action)

                self.hallucinated_replay_buffer.add(
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

                if with_approximate:
                    self.hallucinated_reward_buffer.add(
                        prev_state,
                        prev_action_idx,
                        prev_reward,
                        state,
                        action_idx,
                        reward,
                        next_state,
                        num_action,
                    )

            prev_state = state

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

        if done:
            with_approximate = self.reward_type == "approximate"
            non_zero_elements = [element for element in next_state if element != 0]
            if len(non_zero_elements) > 0:
                self._hallucinate(non_zero_elements, with_approximate)

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
            losses, self.hallucinated_reward_buffer, "hallucinated_step_loss", indices
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
            losses, self.hallucinated_replay_buffer, "hallucinated_loss", indices
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

        self.hallucinated_replay_buffer = ReplayBuffer(
            self.seed,
            self.max_elements,
            1,
            self.hallucinated_buffer_size,
            self.action_dims,
        )
        self.hallucinated_reward_buffer = RewardBuffer(
            self.seed,
            self.step_dims,
            self.max_elements,
            self.estimator_hallucinated_buffer_size,
            self.action_dims,
        )

        super().generate_target_sum(problem_instance)
