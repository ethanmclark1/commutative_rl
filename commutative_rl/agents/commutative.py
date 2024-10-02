import numpy as np

from .utils.agent import Agent


class Commutative(Agent):
    def __init__(
        self,
        seed: int,
        num_agents: int,
        num_large_obstacles: int,
        num_small_obstacles: int,
        config: dict,
    ) -> None:

        super(Commutative, self).__init__(
            seed, num_agents, num_large_obstacles, num_small_obstacles, config
        )

    def _add_to_buffers(
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

        super()._add_to_buffers(
            state, action_idx, reward, next_state, done, episode_step, 0
        )

        if prev_state is None or action_idx == 0:
            self.replay_buffer.increase_size()
            return

        trace_reward = prev_reward + reward

        commutative_state = self.env.get_next_state(prev_state, action_idx)

        transition_1 = (
            prev_state,
            action_idx,
            0,
            commutative_state,
            False,
            episode_step - 1,
            1,
        )
        transition_2 = (
            commutative_state,
            prev_action_idx,
            trace_reward,
            next_state,
            done,
            episode_step,
            2,
        )

        for transition in [transition_1, transition_2]:
            super()._add_to_buffers(*transition)

        self.replay_buffer.increase_size()


class CommutativeWithoutIndices(Commutative):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        noise_type: float,
    ) -> None:

        super(CommutativeWithoutIndices, self).__init__(seed, num_instances, noise_type)

    def _add_to_buffers(
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

        Agent._add_to_buffers(
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
            Agent._add_to_buffers(self, *transition)
            self.replay_buffer.increase_size()
