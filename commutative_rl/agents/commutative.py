import math
import numpy as np

from .utils.agent import Agent
from .utils.helpers import encode, decode


class DoubleTableQTable(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        grid_dims: str,
        n_starts: int,
        n_goals: int,
        n_bridges: int,
        n_episode_steps: int,
        action_success_rate: float,
        utility_scale: float,
        terminal_reward: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
    ) -> None:

        super(DoubleTableQTable, self).__init__(
            seed,
            n_instances,
            grid_dims,
            n_starts,
            n_goals,
            n_bridges,
            n_episode_steps,
            action_success_rate,
            utility_scale,
            terminal_reward,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
        )

    def _setup_problem(self, problem_instance: str) -> None:
        super()._setup_problem(problem_instance)

        self.Q_saa = np.zeros(
            (self.env.n_states, int((self.n_actions + 1) * self.n_actions / 2))
        )

    # treat action pairs as a single action and return paired action_idx
    def _get_paired_idx(self, action_a: int, action_b: int | None) -> int:
        if action_b is None:
            return (self.n_actions * (self.n_actions + 1) // 2) - 1

        # ensure a is always less than b
        a = min(action_a, action_b)
        b = max(action_a, action_b)

        # create triangular matrix to store action pairs
        paired_idx = (self.n_actions * a - (a * (a - 1)) // 2) + (b - a)

        return paired_idx

    def _max_Q_saa(self, state: int, action_idx: int) -> float:
        max_value = -math.inf

        # first action is terminating meaning we cannot pair with any subsequent action
        if self.env.bridge_locations[action_idx] == 0:
            return self.Q_saa[state, self._get_paired_idx(action_idx, None)]

        # iterate through all possible paired actions to return max paired q value
        for i in range(self.env.n_actions):
            index = self._get_paired_idx(action_idx, i)
            if self.Q_saa[state, index] > max_value:
                max_value = self.Q_saa[state, index]

        return max_value

    def _update(
        self,
        state: int,
        action_idx: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
        prev_state: int,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:

        # if episode isn't terminated then add discounted future reward
        if not terminated:
            reward += self.gamma * np.max(self.Q_sa[next_state, :])

        if prev_state is not None:
            pair_idx = self._get_paired_idx(prev_action_idx, action_idx)
            self.Q_saa[prev_state, pair_idx] += self.alpha * (
                prev_reward + reward - self.Q_saa[prev_state, pair_idx]
            )
            self.Q_sa[prev_state, prev_action_idx] = self._max_Q_saa(
                prev_state, prev_action_idx
            )

            # if action is not terminating then update Q_sa with max Q_saa to account for commutative trace
            if self.env.bridge_locations[action_idx] != 0:
                self.Q_sa[prev_state, action_idx] = self._max_Q_saa(
                    prev_state, action_idx
                )

        if terminated:
            pair_idx = self._get_paired_idx(action_idx, None)
            self.Q_saa[state, pair_idx] += self.alpha * (
                reward - self.Q_saa[state, pair_idx]
            )
            self.Q_sa[state, action_idx] = self.Q_saa[state, pair_idx]


class CombinedRewardQTable(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        grid_dims: str,
        n_starts: int,
        n_goals: int,
        n_bridges: int,
        n_episode_steps: int,
        action_success_rate: float,
        utility_scale: float,
        terminal_reward: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
    ) -> None:

        super(CombinedRewardQTable, self).__init__(
            seed,
            n_instances,
            grid_dims,
            n_starts,
            n_goals,
            n_bridges,
            n_episode_steps,
            action_success_rate,
            utility_scale,
            terminal_reward,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
        )

    def _reassign_states(
        self,
        prev_state: int,
        prev_action_idx: int,
        state: int,
        action_idx: int,
        next_state: int,
    ) -> tuple:

        action_a_success = prev_state != state
        action_b_success = state != next_state

        decoded_prev_state = decode(prev_state, self.env.bridge_stages, self.n_bridges)
        decoded_commutative_state = decoded_prev_state.copy()
        decoded_commutative_state[action_idx] = 1
        commutative_state = encode(
            decoded_commutative_state, self.env.bridge_stages, self.n_bridges
        )

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

    def _update(
        self,
        state: int,
        action_idx: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
        prev_state: int,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:

        super()._update(state, action_idx, reward, next_state, terminated, truncated)

        if prev_state is None or self.env.bridge_locations[action_idx] == 0:
            return

        commutative_state, commutative_next_state = self._reassign_states(
            prev_state, prev_action_idx, state, action_idx, next_state
        )

        trace_reward = prev_reward + reward

        transition_1 = (
            prev_state,
            action_idx,
            0,
            commutative_state,
            False,
            False,
        )
        transition_2 = (
            commutative_state,
            prev_action_idx,
            trace_reward,
            commutative_next_state,
            terminated,
            truncated,
        )

        for transition in [transition_1, transition_2]:
            super()._update(*transition)


class HashMapQTable(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        grid_dims: str,
        n_starts: int,
        n_goals: int,
        n_bridges: int,
        n_episode_steps: int,
        action_success_rate: float,
        utility_scale: float,
        terminal_reward: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
    ) -> None:

        super(HashMapQTable, self).__init__(
            seed,
            n_instances,
            grid_dims,
            n_starts,
            n_goals,
            n_bridges,
            n_episode_steps,
            action_success_rate,
            utility_scale,
            terminal_reward,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
        )

        self.transition_map = {}

    def _update(
        self,
        state: int,
        action_idx: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
        prev_state: int,
        prev_action_idx: int,
        prev_reward: float,
    ) -> None:

        super()._update(state, action_idx, reward, next_state, terminated, truncated)

        # Add current transition to transition map
        self.transition_map[(state, action_idx)] = (reward, next_state)

        # Retrieve commutative reward and state from transition map
        commutative_reward, commutative_state = self.transition_map.get(
            (prev_state, action_idx), (None, None)
        )

        if (
            prev_state is None
            or action_idx == 0
            or commutative_reward is None
            or commutative_state is None
        ):
            return

        next_commutative_reward = prev_reward + reward - commutative_reward

        transition_1 = (
            prev_state,
            action_idx,
            commutative_reward,
            commutative_state,
            False,
            False,
        )
        transition_2 = (
            commutative_state,
            prev_action_idx,
            next_commutative_reward,
            next_state,
            terminated,
            truncated,
        )

        for transition in [transition_1, transition_2]:
            super()._update(*transition)
