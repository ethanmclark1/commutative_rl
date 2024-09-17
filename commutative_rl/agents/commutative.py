import copy
import torch
import numpy as np

from .utils.parent import Parent


class Commutative(Parent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: float,
    ) -> None:

        super(Commutative, self).__init__(seed, num_instances, noise_type)

        self.corresponding_index = 0
        self.commutative_replay_buffer = copy.deepcopy(self.replay_buffer)

    def _update_corresponding_index(self) -> None:
        self.corresponding_index = (self.corresponding_index + 1) % self.buffer_size

    def _reassign_states(
        self,
        prev_state: np.ndarray,
        prev_action: int,
        state: np.ndarray,
        action_idx: int,
        next_state: np.ndarray,
    ) -> tuple:

        action_a_success = not np.array_equal(prev_state, state)
        action_b_success = not np.array_equal(state, next_state)

        commutative_state = self.env.place_bridge(prev_state, action_idx)

        if action_a_success and action_b_success:
            pass
        elif not action_a_success and action_b_success:
            if prev_action != action_idx:
                next_state = commutative_state
        elif action_a_success and not action_b_success:
            commutative_state = prev_state
            next_state = state
        else:
            commutative_state = prev_state

        return commutative_state, next_state

    def _add_to_buffers(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        episode_step: int,
        prev_state: np.ndarray = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        super()._add_to_buffers(
            self.replay_buffer,
            state,
            action_idx,
            reward,
            next_state,
            done,
            episode_step,
        )

        if prev_state is None or action_idx == 0:
            self._update_corresponding_index()
            return

        trace_reward = prev_reward + reward

        commutative_state, next_state = self._reassign_states(
            prev_state, prev_action_idx, state, action_idx, next_state
        )

        transition_1 = (
            prev_state,
            action_idx,
            0,
            commutative_state,
            False,
            episode_step - 1,
            self.corresponding_index,
        )
        transition_2 = (
            commutative_state,
            prev_action_idx,
            trace_reward,
            next_state,
            done,
            episode_step,
            self.corresponding_index,
        )

        for transition in [transition_1, transition_2]:
            super()._add_to_buffers(self.commutative_replay_buffer, *transition)

        self._update_corresponding_index()

    def _learn(self, losses: dict) -> None:
        indices = super()._learn(
            losses, self.replay_buffer, loss_type="traditional_loss"
        )

        if indices is None:
            return

        corresponding_indices = self.commutative_replay_buffer.corresponding_indices
        mask = corresponding_indices.unsqueeze(-1) == indices
        commutative_indices = mask.nonzero()[:, 0]
        commutative_indices = commutative_indices[
            torch.randperm(commutative_indices.size(0))
        ]

        super()._learn(
            losses,
            self.commutative_replay_buffer,
            commutative_indices,
            "commutative_loss",
        )
