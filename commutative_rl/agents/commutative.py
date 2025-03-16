import math
import copy
import torch
import numpy as np

from .utils.agent import Agent
from .utils.networks import MLP
from .utils.buffers import ReplayBuffer
from .utils.helpers import encode, decode


# Reassign states to account for commutative trace (CombinedReward approaches)
def reassign_states(
    prev_state: float,
    prev_action_idx: int,
    state: float,
    action_idx: int,
    next_state: float,
    n_bridges: int,
    bridge_stages: int,
    n_states: int,
) -> tuple:

    action_a_success = prev_state != state
    action_b_success = state != next_state

    decoded_prev_state = decode(prev_state, bridge_stages, n_bridges, n_states)
    decoded_commutative_state = decoded_prev_state.copy()
    decoded_commutative_state[action_idx] = 1
    commutative_state = encode(
        decoded_commutative_state, bridge_stages, n_bridges, n_states
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


"""
Exact Approaches
----------------
"""


class SuperActionQTable(Agent):
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
        early_termination_penalty: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
    ) -> None:

        super(SuperActionQTable, self).__init__(
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
            early_termination_penalty,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
        )

    def _setup_problem(self, problem_instance: str) -> None:
        Agent._setup_problem(self, problem_instance)
        self.Q_sa = np.zeros((self.n_states, self.n_actions))
        self.Q_sab = np.zeros(
            (
                self.n_states,
                int((self.n_actions + 1) * self.n_actions / 2),
            ),
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

    def _max_Q_sab(self, state: int, action_idx: int) -> float:
        max_value = -math.inf

        # first action is terminating meaning we cannot pair with any subsequent action
        if self.env.bridge_locations[action_idx] == 0:
            return self.Q_sab[state, self._get_paired_idx(action_idx, None)]

        # iterate through all possible paired actions to return max paired q value
        for i in range(self.env.n_actions):
            index = self._get_paired_idx(action_idx, i)
            if self.Q_sab[state, index] > max_value:
                max_value = self.Q_sab[state, index]

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

        state = int(state * self.n_states)
        next_state = int(next_state * self.n_states)

        # if episode isn't terminated then add discounted future reward
        if not terminated:
            reward += self.gamma * np.max(self.Q_sa[next_state, :])

        if prev_state is not None:
            prev_state = int(prev_state * self.n_states)
            pair_idx = self._get_paired_idx(prev_action_idx, action_idx)
            self.Q_sab[prev_state, pair_idx] += self.alpha * (
                prev_reward + reward - self.Q_sab[prev_state, pair_idx]
            )
            self.Q_sa[prev_state, prev_action_idx] = self._max_Q_sab(
                prev_state, prev_action_idx
            )

            # if action is not terminating then update Q_sa with max Q_saa to account for commutative trace
            if self.env.bridge_locations[action_idx] != 0:
                self.Q_sa[prev_state, action_idx] = self._max_Q_sab(
                    prev_state, action_idx
                )

        if terminated:
            pair_idx = self._get_paired_idx(action_idx, None)
            self.Q_sab[state, pair_idx] += self.alpha * (
                reward - self.Q_sab[state, pair_idx]
            )
            self.Q_sa[state, action_idx] = self.Q_sab[state, pair_idx]


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
        early_termination_penalty: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
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
            early_termination_penalty,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
        )

    def _setup_problem(self, problem_instance: str) -> None:
        Agent._setup_problem(self, problem_instance)
        self.Q_sa = np.zeros((self.n_states, self.n_actions))

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

        Agent._update(
            self, state, action_idx, reward, next_state, terminated, truncated
        )

        if prev_state is None or self.env.bridge_locations[action_idx] == 0:
            return

        commutative_state, commutative_next_state = reassign_states(
            prev_state,
            prev_action_idx,
            state,
            action_idx,
            next_state,
            self.env.n_bridges,
            self.env.bridge_stages,
            self.n_states,
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
            Agent._update(self, *transition)


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
        early_termination_penalty: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
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
            early_termination_penalty,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
        )

    def _setup_problem(self, problem_instance) -> None:
        Agent._setup_problem(self, problem_instance)
        self.Q_sa = np.zeros((self.n_states, self.n_actions))
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

        Agent._update(
            self, state, action_idx, reward, next_state, terminated, truncated
        )

        if prev_state is None or self.env.bridge_locations[action_idx] == 0:
            return

        # Add current transition to transition map
        denormalized_state = int(state * self.n_states)
        self.transition_map[(denormalized_state, action_idx)] = (
            reward,
            next_state,
        )

        # Retrieve commutative reward and commutative state from transition map
        denormalized_prev_state = int(prev_state * self.n_states)
        commutative_reward, commutative_state = self.transition_map.get(
            (denormalized_prev_state, action_idx), (None, None)
        )

        if commutative_reward is None or commutative_state is None:
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
            Agent._update(self, *transition)


"""
Approximate Approaches
----------------------
"""


class OnlineSuperActionDQN(Agent):
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
        early_termination_penalty: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
    ) -> None:

        super(OnlineSuperActionDQN, self).__init__(
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
            early_termination_penalty,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
        )

    def _setup_problem(self, problem_instance) -> None:
        Agent._setup_problem(self, problem_instance)

        output_dims = (self.n_actions + 1) * self.n_actions // 2
        self.network = MLP(
            self.seed,
            self.state_dims,
            output_dims,
            self.hidden_dims,
            self.n_hidden_layers,
            self.alpha,
            self.dropout,
        ).to(self.device)

        self.target_network = copy.deepcopy(self.network)
        self.buffer = ReplayBuffer(
            self.seed,
            self.state_dims,
            self.batch_size,
            self.buffer_size,
            self.device,
        )

        self.network.train()
        self.target_network.eval()

    def _greedy_policy(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            action_idx = self._evaluate(state)[1].item()

        return action_idx

    # treat action pairs as a single action and return paired action_idx
    def _get_paired_idx(
        self, action_a: int | torch.Tensor, action_b: int | torch.Tensor | None
    ) -> torch.Tensor:

        if not isinstance(action_a, torch.Tensor):
            action_a = torch.as_tensor(action_a, device=self.device)
        if not isinstance(action_b, torch.Tensor):
            action_b = torch.as_tensor(action_b, device=self.device)

        none_mask = action_b == -1  # -1 represents None

        a = torch.min(action_a, action_b)
        b = torch.max(action_a, action_b)

        indices = (self.n_actions * a - (a * (a - 1)) // 2) + (b - a)
        indices[none_mask] = (self.n_actions * (self.n_actions + 1) // 2) - 1

        return indices

    def _add_to_buffer(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        terminated: bool,
        truncated: bool,
        prev_state: float = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        if prev_state is not None:
            paired_idx = self._get_paired_idx(prev_action_idx, action_idx)
            commutative_reward = prev_reward + reward
            self.buffer.add(
                prev_state,
                paired_idx,
                commutative_reward,
                next_state,
                terminated,
            )
            self._online_learn()

            if self.env.bridge_locations[action_idx] == 0:
                paired_idx = self._get_paired_idx(action_idx, -1)
                self.buffer.add(state, paired_idx, reward, next_state, terminated)
                self._online_learn()

    # return max paired Q-value for each action and corresponding paired action to take
    def _max_Q_saa(
        self, current_paired_q_vals: torch.Tensor, action_idxs: torch.Tensor
    ) -> tuple:

        # get position of terminating action
        termination_mask = action_idxs == (self.n_actions - 1)
        # get paired index with terminating action as first action
        termination_idx = self._get_paired_idx(
            action_idxs[termination_mask],
            torch.full_like(action_idxs[termination_mask], -1),
        )
        termination_q_val = current_paired_q_vals[0, termination_idx]

        nonterminating_actions = action_idxs[~termination_mask]
        # get all possible next actions that can be paired with current actions
        all_possible_next = torch.arange(self.n_actions).to(self.device)

        # generate all possible permutations of actions
        action_pairs = torch.cartesian_prod(nonterminating_actions, all_possible_next)
        paired_indices = self._get_paired_idx(action_pairs[:, 0], action_pairs[:, 1])

        # reshape paired Q-values to be (nonterminating_actions, n_actions)
        paired_q_vals = current_paired_q_vals[0][paired_indices].reshape(
            nonterminating_actions.shape[0], self.n_actions
        )
        # get max Q value for each first action
        max_paired_q_vals_no_termination, best_next_actions = torch.max(
            paired_q_vals, axis=1
        )

        # add back in paired Q-values for terminating action
        max_paired_q_vals = torch.zeros_like(action_idxs, dtype=torch.float32)
        max_paired_q_vals[termination_mask] = termination_q_val
        max_paired_q_vals[~termination_mask] = max_paired_q_vals_no_termination

        # best second actions to take based on first action
        next_action_idxs = torch.full_like(action_idxs, 0)
        next_action_idxs[~termination_mask] = best_next_actions

        return max_paired_q_vals, next_action_idxs

    # given a state, return max paired Q-value, action to take, and next action to take
    def _evaluate(self, state: torch.Tensor) -> tuple:
        current_paired_q_vals = self.target_network(state)

        action_idxs = torch.arange(self.n_actions).to(self.device)

        # returns the max paired Q value and which second action to take to achieve that
        max_paired_q_vals, next_action_idxs = self._max_Q_saa(
            current_paired_q_vals, action_idxs
        )

        best_idx = torch.argmax(max_paired_q_vals)

        max_paired_q_val = max_paired_q_vals[best_idx].item()
        action_idx = action_idxs[best_idx].item()
        next_action_idx = next_action_idxs[best_idx].item()

        return max_paired_q_val, action_idx, next_action_idx

    def _online_learn(self) -> None:
        if self.buffer.real_size < self.batch_size:
            return

        states, action_idxs, rewards, next_states, terminations = self.buffer.sample()

        for i in range(self.batch_size):
            with torch.no_grad():
                max_next_q_value = self._evaluate(next_states[i])[0]
                target = rewards[i] + self.gamma * ~terminations[i] * max_next_q_value

                target_q_values = self.target_network(states[i])
                target_q_values[0, action_idxs[i]] = target

            current_q_values = self.network(states[i])

            self.network.optimizer.zero_grad()
            loss = self.network.loss_fn(current_q_values, target_q_values)
            loss.backward()
            self.network.optimizer.step()


class OfflineSuperActionDQN(OnlineSuperActionDQN):
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
        early_termination_penalty: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
    ) -> None:

        super(OfflineSuperActionDQN, self).__init__(
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
            early_termination_penalty,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
        )

    def _add_to_buffer(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        terminated: bool,
        truncated: bool,
        prev_state: float = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        if prev_state is not None:
            paired_idx = self._get_paired_idx(prev_action_idx, action_idx)
            commutative_reward = prev_reward + reward
            self.buffer.add(
                prev_state,
                paired_idx,
                commutative_reward,
                next_state,
                terminated,
            )

            if self.env.bridge_locations[action_idx] == 0:
                paired_idx = self._get_paired_idx(action_idx, -1)
                self.buffer.add(state, paired_idx, reward, next_state, terminated)

    def _offline_learn(self):
        if self.buffer.real_size < self.batch_size:
            return

        states, action_idxs, rewards, next_states, terminations = self.buffer.sample()

        q_values = self.network(states)
        current_q_values = q_values.gather(1, action_idxs)

        with torch.no_grad():
            max_next_q_value = self._evaluate(next_states)[0]
            target_q_values = rewards + self.gamma * ~terminations * max_next_q_value

        self.network.optimizer.zero_grad()
        loss = self.network.loss_fn(current_q_values, target_q_values)
        loss.backward()
        self.network.optimizer.step()


class OnlineCombinedRewardDQN(Agent):
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
        early_termination_penalty: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
    ) -> None:

        super(OnlineCombinedRewardDQN, self).__init__(
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
            early_termination_penalty,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
        )

    def _setup_problem(self, problem_instance) -> None:
        Agent._setup_problem(self, problem_instance)

        self.network = MLP(
            self.seed,
            self.state_dims,
            self.n_actions,
            self.hidden_dims,
            self.n_hidden_layers,
            self.alpha,
            self.dropout,
        ).to(self.device)

        self.target_network = copy.deepcopy(self.network)
        self.buffer = ReplayBuffer(
            self.seed,
            self.state_dims,
            self.batch_size,
            self.buffer_size,
            self.device,
        )

        self.network.train()
        self.target_network.eval()

    def _add_to_buffer(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        terminated: bool,
        truncated: bool,
        prev_state: float = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        Agent._add_to_buffer(self, state, action_idx, reward, next_state, terminated)
        Agent._online_learn(self)

        if prev_state is None or self.env.bridge_locations[action_idx] == 0:
            return

        commutative_state, commutative_next_state = reassign_states(
            prev_state,
            prev_action_idx,
            state,
            action_idx,
            next_state,
            self.env.n_bridges,
            self.env.bridge_stages,
            self.n_states,
        )

        trace_reward = prev_reward + reward

        transition_1 = (
            prev_state,
            action_idx,
            0,
            commutative_state,
            False,
        )
        transition_2 = (
            commutative_state,
            prev_action_idx,
            trace_reward,
            commutative_next_state,
            terminated,
        )

        for transition in [transition_1, transition_2]:
            Agent._add_to_buffer(self, *transition)
            Agent._online_learn(self)


class OfflineCombinedRewardDQN(OnlineCombinedRewardDQN):
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
        early_termination_penalty: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
    ) -> None:

        super(OfflineCombinedRewardDQN, self).__init__(
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
            early_termination_penalty,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
        )

    def _add_to_buffer(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        terminated: bool,
        truncated: bool,
        prev_state: float = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        Agent._add_to_buffer(self, state, action_idx, reward, next_state, terminated)

        if prev_state is None or self.env.bridge_locations[action_idx] == 0:
            return

        commutative_state, commutative_next_state = reassign_states(
            prev_state,
            prev_action_idx,
            state,
            action_idx,
            next_state,
            self.env.n_bridges,
            self.env.bridge_stages,
            self.n_states,
        )

        trace_reward = prev_reward + reward

        transition_1 = (prev_state, action_idx, 0, commutative_state, False)
        transition_2 = (
            commutative_state,
            prev_action_idx,
            trace_reward,
            commutative_next_state,
            terminated,
        )

        for transition in [transition_1, transition_2]:
            Agent._add_to_buffer(self, *transition)


class OnlineHashMapDQN(Agent):
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
        early_termination_penalty: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
    ) -> None:

        super(OnlineHashMapDQN, self).__init__(
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
            early_termination_penalty,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
        )

    def _setup_problem(self, problem_instance) -> None:
        Agent._setup_problem(self, problem_instance)

        self.transition_map = {}

        self.network = MLP(
            self.seed,
            self.state_dims,
            self.n_actions,
            self.hidden_dims,
            self.n_hidden_layers,
            self.alpha,
            self.dropout,
        ).to(self.device)

        self.target_network = copy.deepcopy(self.network)
        self.buffer = ReplayBuffer(
            self.seed,
            self.state_dims,
            self.batch_size,
            self.buffer_size,
            self.device,
        )

        self.network.train()
        self.target_network.eval()

    def _add_to_buffer(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        terminated: bool,
        truncated: bool,
        prev_state: float = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        Agent._add_to_buffer(self, state, action_idx, reward, next_state, terminated)
        Agent._online_learn(self)

        if prev_state is None or self.env.bridge_locations[action_idx] == 0:
            return

        # Add current transition to transition map
        denormalized_state = int(state * self.env.n_states)
        denormalized_next_state = int(next_state * self.env.n_states)
        self.transition_map[(denormalized_state, action_idx)] = (
            reward,
            denormalized_next_state,
        )

        # Retrieve commutative reward and state from transition map
        denormalized_prev_state = int(prev_state * self.env.n_states)
        commutative_reward, denormalized_commutative_state = self.transition_map.get(
            (denormalized_prev_state, action_idx), (None, None)
        )

        if commutative_reward is None or denormalized_commutative_state is None:
            return

        commutative_state = denormalized_commutative_state / self.env.n_states
        next_commutative_reward = prev_reward + reward - commutative_reward

        transition_1 = (
            prev_state,
            action_idx,
            commutative_reward,
            commutative_state,
            False,
        )
        transition_2 = (
            commutative_state,
            prev_action_idx,
            next_commutative_reward,
            next_state,
            terminated,
        )

        for transition in [transition_1, transition_2]:
            Agent._add_to_buffer(self, *transition)
            Agent._online_learn(self)


class OfflineHashMapDQN(OnlineHashMapDQN):
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
        early_termination_penalty: int,
        duplicate_bridge_penalty: float,
        alpha: float = None,
        epsilon: float = None,
        gamma: float = None,
        batch_size: int = None,
        buffer_size: int = None,
        hidden_dims: int = None,
        n_hidden_layers: int = None,
        target_update_freq: int = None,
        dropout: float = None,
    ) -> None:

        super(OfflineHashMapDQN, self).__init__(
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
            early_termination_penalty,
            duplicate_bridge_penalty,
            alpha,
            epsilon,
            gamma,
            batch_size,
            buffer_size,
            hidden_dims,
            n_hidden_layers,
            target_update_freq,
            dropout,
        )

    def _add_to_buffer(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        terminated: bool,
        truncated: bool,
        prev_state: float = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        Agent._add_to_buffer(self, state, action_idx, reward, next_state, terminated)

        if prev_state is None or self.env.bridge_locations[action_idx] == 0:
            return

        # Add current transition to transition map
        denormalized_state = int(state * self.env.n_states)
        denormalized_next_state = int(next_state * self.env.n_states)
        self.transition_map[(denormalized_state, action_idx)] = (
            reward,
            denormalized_next_state,
        )

        # Retrieve commutative reward and state from transition map
        denormalized_prev_state = int(prev_state * self.env.n_states)
        commutative_reward, denormalized_commutative_state = self.transition_map.get(
            (denormalized_prev_state, action_idx), (None, None)
        )

        if commutative_reward is None or denormalized_commutative_state is None:
            return

        commutative_state = denormalized_commutative_state / self.env.n_states
        next_commutative_reward = prev_reward + reward - commutative_reward

        transition_1 = (
            prev_state,
            action_idx,
            commutative_reward,
            commutative_state,
            False,
        )
        transition_2 = (
            commutative_state,
            prev_action_idx,
            next_commutative_reward,
            next_state,
            terminated,
        )

        for transition in [transition_1, transition_2]:
            Agent._add_to_buffer(self, *transition)
