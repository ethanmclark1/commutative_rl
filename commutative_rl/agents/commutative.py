import math
import copy
import torch
import numpy as np

from .utils.agent import Agent
from .utils.networks import MLP
from .utils.buffers import ReplayBuffer


# Reassign states to account for commutative trace (CombinedReward approaches)
def reassign_states(prev_state: float, state: float, next_state: float) -> tuple:
    action_a = state - prev_state
    action_b = next_state - state

    commutative_state = prev_state + action_b
    commutative_next_state = commutative_state + action_a

    return commutative_state, commutative_next_state


"""
Exact Approaches
----------------
"""


class SuperActionQTable(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        step_value: float = None,
        over_penalty: float = None,
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
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            step_value,
            over_penalty,
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
            )
        )

    # treat action pairs as a super action and return super action_idx
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
        if self.env.elements[action_idx] == 0:
            return self.Q_sab[state, self._get_paired_idx(action_idx, None)]

        # iterate through all possible paired actions to return max paired q value
        for i in range(self.env.n_actions):
            index = self._get_paired_idx(action_idx, i)
            if self.Q_sab[state, index] > max_value:
                max_value = self.Q_sab[state, index]

        return max_value

    def _update(
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

        state = int(state * self.env.target_sum)
        next_state = int(next_state * self.env.target_sum)

        # If episode isn't terminated then add discounted future reward
        if not terminated:
            reward += self.gamma * np.max(self.Q_sa[next_state, :])

        if prev_state is not None:
            prev_state = int(prev_state * self.env.target_sum)

            pair_idx = self._get_paired_idx(prev_action_idx, action_idx)
            self.Q_sab[prev_state, pair_idx] += self.alpha * (
                prev_reward + reward - self.Q_sab[prev_state, pair_idx]
            )
            self.Q_sa[prev_state, prev_action_idx] = self._max_Q_sab(
                prev_state, prev_action_idx
            )

            # if action is not terminating then update Q_sa with max Q_sab to account for commutative trace
            if self.env.elements[action_idx] != 0:
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
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        step_value: float = None,
        over_penalty: float = None,
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
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            step_value,
            over_penalty,
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
        prev_state: int = None,
        prev_action_idx: int = None,
        prev_reward: float = None,
    ) -> None:

        Agent._update(
            self, state, action_idx, reward, next_state, terminated, truncated
        )

        if prev_state is None or self.env.elements[action_idx] == 0:
            return

        commutative_state, commutative_next_state = reassign_states(
            prev_state, state, next_state
        )

        trace_reward = prev_reward + reward

        transition_1 = (
            prev_state,
            action_idx,
            0,
            commutative_state,
            commutative_state >= 2.0,  # Terminated if over sum limit
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
            # Skip transitions where state is over the normalized sum limit
            if transition[0] >= 2.0:
                continue

            Agent._update(self, *transition)


class HashMapQTable(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        step_value: float = None,
        over_penalty: float = None,
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
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            step_value,
            over_penalty,
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

        Agent._update(
            self, state, action_idx, reward, next_state, terminated, truncated
        )

        if prev_state is None or self.env.elements[action_idx] == 0:
            return

        # Add current transition to transition map
        denormalized_state = int(state * self.env.target_sum)
        self.transition_map[(denormalized_state, action_idx)] = (
            reward,
            next_state,
        )

        # Retrieve commutative reward and commutative state from transition map
        denormalized_prev_state = int(prev_state * self.env.target_sum)
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
            commutative_state >= 2.0,  # Terminated if over sum limit
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
            # Skip transitions where state is over the normalized sum limit
            if transition[0] >= 2.0:
                continue

            Agent._update(self, *transition)


"""
Approximate Approaches
----------------------
"""


class SuperActionDQN(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        step_value: float = None,
        over_penalty: float = None,
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

        super(SuperActionDQN, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            step_value,
            over_penalty,
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

    # treat action pairs as a single action and return paired action_idx
    def _get_paired_idx(
        self, action_a: int | torch.Tensor, action_b: int | torch.Tensor | None
    ) -> torch.Tensor:

        if isinstance(action_a, int):
            action_a = torch.as_tensor(action_a, device=self.device)
        if isinstance(action_b, int):
            action_b = torch.as_tensor(action_b, device=self.device)

        none_mask = action_b == -1  # -1 represents None

        a = torch.min(action_a, action_b)
        b = torch.max(action_a, action_b)

        indices = (self.n_actions * a - (a * (a - 1)) // 2) + (b - a)
        indices[none_mask] = (self.n_actions * (self.n_actions + 1) // 2) - 1

        return indices

    def _greedy_policy(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            action_idx = self._evaluate(state)[1]

        return action_idx

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

            if self.env.elements[action_idx] == 0:
                paired_idx = self._get_paired_idx(action_idx, -1)
                self.buffer.add(state, paired_idx, reward, next_state, terminated)

    # return max paired Q-value for each action and corresponding paired action to take
    def _max_Q_sab(
        self, current_paired_q_vals: torch.Tensor, action_idxs: torch.Tensor
    ) -> tuple:

        # get position of terminating action
        termination_mask = action_idxs == (self.n_actions - 1)
        # get paired index with terminating action as first action
        termination_idx = self._get_paired_idx(
            action_idxs[termination_mask],
            torch.full_like(action_idxs[termination_mask], -1),
        ).item()
        termination_q_val = current_paired_q_vals[:, termination_idx].reshape(-1, 1)

        nonterminating_actions = action_idxs[~termination_mask]
        # get all possible next actions that can be paired with current actions
        all_possible_next = torch.arange(self.n_actions).to(self.device)

        # generate all possible permutations of actions
        action_pairs = torch.cartesian_prod(nonterminating_actions, all_possible_next)
        paired_indices = self._get_paired_idx(action_pairs[:, 0], action_pairs[:, 1])

        # reshape paired Q-values to be (batch_size, nonterminating_actions, n_elems)
        paired_q_vals = current_paired_q_vals[:, paired_indices].reshape(
            current_paired_q_vals.shape[0],
            nonterminating_actions.shape[0],
            self.n_actions,
        )
        # get max Q value for each first action
        max_paired_q_vals_no_termination = torch.max(paired_q_vals, axis=2)[0]

        # add back in paired Q-values for terminating action
        max_paired_q_vals = torch.zeros(
            current_paired_q_vals.shape[0], action_idxs.shape[0], device=self.device
        )
        max_paired_q_vals[:, termination_mask] = termination_q_val
        max_paired_q_vals[:, ~termination_mask] = max_paired_q_vals_no_termination

        return max_paired_q_vals

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

        # reshape paired Q-values to be (nonterminating_actions, n_elems)
        paired_q_vals = current_paired_q_vals[0][paired_indices].reshape(
            nonterminating_actions.shape[0], self.n_actions
        )
        # get max Q value for each first action
        max_paired_q_vals_no_termination = torch.max(paired_q_vals, axis=1)[0]

        # add back in paired Q-values for terminating action
        max_paired_q_vals = torch.zeros_like(action_idxs, dtype=torch.float32)
        max_paired_q_vals[termination_mask] = termination_q_val
        max_paired_q_vals[~termination_mask] = max_paired_q_vals_no_termination

        return max_paired_q_vals

    # given a state, return max paired Q-value and action to take that leads to max Q-value
    def _evaluate(self, state: torch.Tensor) -> tuple:
        current_paired_q_vals = self.target_network(state)

        action_idxs = torch.arange(self.n_actions).to(self.device)

        # returns the max paired Q value given the current paired Q values
        max_paired_q_vals = self._max_Q_sab(current_paired_q_vals, action_idxs)
        if state.shape[0] == 1:
            max_paired_cheese = self._max_Q_saa(current_paired_q_vals, action_idxs)
            assert torch.all(max_paired_q_vals == max_paired_cheese)

        best_idx = torch.argmax(max_paired_q_vals, dim=1).unsqueeze(1)

        max_paired_q_val = max_paired_q_vals.gather(1, best_idx)
        action_idx = action_idxs[best_idx].squeeze(1)

        return max_paired_q_val, action_idx

    def _online_evaluate(self, state: torch.Tensor) -> tuple:
        current_paired_q_vals = self.target_network(state)

        action_idxs = torch.arange(self.n_actions).to(self.device)

        # returns the max paired Q value and which second action to take to achieve that
        max_paired_q_vals = self._max_Q_saa(current_paired_q_vals, action_idxs)

        best_idx = torch.argmax(max_paired_q_vals)

        max_paired_q_val = max_paired_q_vals[best_idx].item()
        action_idx = action_idxs[best_idx].item()

        return max_paired_q_val, action_idx

    def _learn(self):
        if self.buffer.real_size < self.batch_size:
            return

        states, action_idxs, rewards, next_states, terminations = self.buffer.sample()

        q_values = self.network(states)
        current_q_values = q_values.gather(1, action_idxs)

        with torch.no_grad():
            max_next_q_value = self._evaluate(next_states)[0]

            for i in range(self.batch_size):
                cheese = self._online_evaluate(next_states[i].unsqueeze(0))[0]
                assert torch.allclose(
                    max_next_q_value[i], torch.tensor([cheese], device=self.device)
                )

            target_q_values = rewards + self.gamma * ~terminations * max_next_q_value

        self.network.optimizer.zero_grad()
        loss = self.network.loss_fn(current_q_values, target_q_values)
        loss.backward()
        self.network.optimizer.step()


class CombinedRewardDQN(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        step_value: float = None,
        over_penalty: float = None,
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

        super(CombinedRewardDQN, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            step_value,
            over_penalty,
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

        if prev_state is None or self.env.elements[action_idx] == 0:
            return

        commutative_state, commutative_next_state = reassign_states(
            prev_state, state, next_state
        )

        trace_reward = prev_reward + reward

        transition_1 = (
            prev_state,
            action_idx,
            0,
            commutative_state,
            commutative_state >= self.env.sum_limit,
        )
        transition_2 = (
            commutative_state,
            prev_action_idx,
            trace_reward,
            commutative_next_state,
            terminated,
        )

        for transition in [transition_1, transition_2]:
            # Skip transitions where state is over the sum
            if transition[0] >= self.env.sum_limit:
                continue

            Agent._add_to_buffer(self, *transition)


class HashMapDQN(Agent):
    def __init__(
        self,
        seed: int,
        n_instances: int,
        sum_range: range,
        elem_range: range,
        n_actions: int,
        max_noise: float,
        step_value: float = None,
        over_penalty: float = None,
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

        super(HashMapDQN, self).__init__(
            seed,
            n_instances,
            sum_range,
            elem_range,
            n_actions,
            max_noise,
            step_value,
            over_penalty,
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

        if prev_state is None or self.env.elements[action_idx] == 0:
            return

        # Add current transition to transition map
        denormalized_state = int(state * self.env.target_sum)
        denormalized_next_state = int(next_state * self.env.target_sum)
        self.transition_map[(denormalized_state, action_idx)] = (
            reward,
            denormalized_next_state,
        )

        # Retrieve commutative reward and state from transition map
        denormalized_prev_state = int(prev_state * self.env.target_sum)
        commutative_reward, denormalized_commutative_state = self.transition_map.get(
            (denormalized_prev_state, action_idx), (None, None)
        )

        if commutative_reward is None or denormalized_commutative_state is None:
            return

        commutative_state = denormalized_commutative_state / self.env.target_sum
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
