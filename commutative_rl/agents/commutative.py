import numpy as np

from .utils.agent import Agent


class Commutative(Agent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
        alpha: float,
        buffer_size: int,
        target_update_freq: int,
    ) -> None:

        super(Commutative, self).__init__(
            seed,
            num_instances,
            noise_type,
            alpha,
            buffer_size,
            target_update_freq,
        )

    def _reassign_states(
        self,
        prev_state: np.ndarray,
        prev_action_idx: int,
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
            if prev_action_idx != action_idx:
                next_state = commutative_state
        elif action_a_success and not action_b_success:
            commutative_state = prev_state
            next_state = state
        else:
            commutative_state = prev_state

        return commutative_state, next_state

    def _add_to_buffer(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        episode_step: int,
        prev_state: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:

        super()._add_to_buffer(
            state,
            action_idx,
            reward,
            next_state,
            done,
            episode_step,
        )

        if prev_state is None or action_idx == 0:
            self.replay_buffer.increase_size()
            return

        # average_reward = (prev_reward + reward) / 2
        # commutative_reward = average_reward + (self.action_cost / 2)
        # next_commutative_reward = average_reward - (self.action_cost / 2)

        commutative_state, next_state = self._reassign_states(
            prev_state, prev_action_idx, state, action_idx, next_state
        )

        commutative_reward = self.env._get_reward(
            prev_state, commutative_state, False, episode_step - 1
        )
        next_commutative_reward = self.env._get_reward(
            commutative_state, next_state, done, episode_step
        )

        # commutative_state, commutative_reward, prev_done = self.env.step(prev_state, action_idx, episode_step - 1)
        # next_commutative_state, next_commutative_reward, done = self.env.step(commutative_state, prev_action_idx, episode_step)

        transition_1 = (
            prev_state,
            action_idx,
            commutative_reward,
            commutative_state,
            False,
            episode_step - 1,
            1,
        )

        transition_2 = (
            commutative_state,
            prev_action_idx,
            next_commutative_reward,
            next_state,
            done,
            episode_step,
            2,
        )

        for transition in [transition_1, transition_2]:
            super()._add_to_buffer(*transition)

        self.replay_buffer.increase_size()


class TripleCommutative(Commutative):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
        alpha: float,
        buffer_size: int,
        target_update_freq: int,
    ) -> None:

        super(TripleCommutative, self).__init__(
            seed, num_instances, noise_type, alpha, buffer_size, target_update_freq
        )

        self.batch_size *= 3
        self.buffer_size *= 3

        self.config["dqn"]["batch_size"] = self.batch_size
        self.config["dqn"]["buffer_size"] = self.buffer_size


class CommutativeFullBatch(Commutative):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
        alpha: float,
        buffer_size: int,
        target_update_freq: int,
    ) -> None:

        super(CommutativeFullBatch, self).__init__(
            seed,
            num_instances,
            noise_type,
            alpha,
            buffer_size,
            target_update_freq,
        )


class CommutativeWithoutIndices(Commutative):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: str,
        alpha: float,
        buffer_size: int,
        target_update_freq: int,
    ) -> None:

        super(CommutativeWithoutIndices, self).__init__(
            seed,
            num_instances,
            noise_type,
            alpha,
            buffer_size,
            target_update_freq,
        )

    def _add_to_buffer(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        episode_step: int,
        prev_state: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:

        Agent._add_to_buffer(
            self, state, action_idx, reward, next_state, done, episode_step
        )

        self.replay_buffer.increase_size()

        if prev_state is None or action_idx == 0:
            return

        trace_reward = prev_reward + reward

        commutative_state, next_state = Commutative._reassign_states(
            self, prev_state, prev_action_idx, state, action_idx, next_state
        )

        transition_1 = (
            prev_state,
            action_idx,
            0,
            commutative_state,
            False,
            episode_step - 1,
        )
        transition_2 = (
            commutative_state,
            prev_action_idx,
            trace_reward,
            next_state,
            done,
            episode_step,
        )

        for transition in [transition_1, transition_2]:
            Agent._add_to_buffer(self, *transition)
            self.replay_buffer.increase_size()
