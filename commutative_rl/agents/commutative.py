import copy
import torch

from .utils.agent import Agent
from .utils.networks import DQN
from .utils.buffers import ReplayBuffer


class CommutativeDQN(Agent):
    def __init__(
        self,
        seed: int,
        n_agents: int,
        n_large_obstacles: int,
        n_small_obstacles: int,
        n_episode_steps: int,
        granularity: float,
        safe_area_multiplier: float,
        failed_path_cost: float,
        configs_to_consider: int,
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

        super(CommutativeDQN, self).__init__(
            seed,
            n_agents,
            n_large_obstacles,
            n_small_obstacles,
            n_episode_steps,
            granularity,
            safe_area_multiplier,
            failed_path_cost,
            configs_to_consider,
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

        output_dims = (self.n_actions + 1) * self.n_actions // 2
        self.network = DQN(
            seed,
            self.state_dims,
            output_dims,
            self.hidden_dims,
            self.n_hidden_layers,
            self.alpha,
            self.dropout,
        ).to(self.device)

        self.target_network = copy.deepcopy(self.network)
        self.buffer = ReplayBuffer(
            seed, self.state_dims, self.batch_size, self.buffer_size, self.device
        )

        self.network.train()
        self.target_network.eval()

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

        if not terminated:
            if prev_state is not None:
                paired_idx = self._get_paired_idx(prev_action_idx, action_idx)
                commutative_reward = prev_reward + reward
                self.buffer.add(
                    prev_state, paired_idx, commutative_reward, next_state, terminated
                )
                self._learn()
        elif terminated or truncated:
            if self.env.candidate_lines[action_idx] == 0:
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
                    self._learn()
                    paired_idx = self._get_paired_idx(action_idx, -1)
                    self.buffer.add(state, paired_idx, reward, next_state, terminated)
                    self._learn()
            else:
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
                    self._learn()

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

    def _learn(self) -> None:
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
