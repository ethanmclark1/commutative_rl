import torch
import numpy as np

from .helpers import *


class ReplayBuffer:
    def __init__(
        self,
        seed: int,
        n_cells: int,
        n_steps: int,
        buffer_size: int,
    ) -> None:

        self.n_cells = n_cells
        self.n_steps = n_steps

        self.states = torch.zeros(buffer_size, 3, n_cells)
        self.action_idxs = torch.zeros(buffer_size, 3, 1, dtype=torch.int64)
        self.rewards = torch.zeros(buffer_size, 3, 1)
        self.next_states = torch.zeros(buffer_size, 3, n_cells)
        self.dones = torch.zeros(buffer_size, 3, 1, dtype=torch.bool)
        self.episode_steps = torch.zeros(buffer_size, 3, 1)
        self.initialized = torch.zeros(buffer_size, 3, 1, dtype=torch.bool)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.sample_rng = torch.Generator().manual_seed(seed)

    def increase_size(self) -> None:
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def add(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        episode_step: int,
        row: int,
    ) -> None:

        episode_step = encode(episode_step, self.n_steps)

        self.states[self.count][row] = torch.as_tensor(state)
        self.action_idxs[self.count][row] = torch.as_tensor(action_idx)
        self.rewards[self.count][row] = torch.as_tensor(reward)
        self.next_states[self.count][row] = torch.as_tensor(next_state)
        self.dones[self.count][row] = torch.as_tensor(done)
        self.episode_steps[self.count][row] = torch.as_tensor(episode_step)
        self.initialized[self.count][row] = torch.as_tensor(True)

    def sample(self, name: str, batch_size: int) -> tuple:
        valid_indices = torch.arange(self.real_size)
        random_indices = torch.randint(
            0, valid_indices.size(0), (batch_size,), generator=self.sample_rng
        )

        sampled_indices = valid_indices[random_indices]
        initialized_mask = self.initialized[sampled_indices]

        row_counts = initialized_mask.sum(dim=1)
        cumulative_rows = torch.cumsum(row_counts, dim=0)
        last_index = torch.where(cumulative_rows <= batch_size)[0][-1].item()
        indices_range = list(range(last_index + 1))

        if name == "CommutativeFullBatch":
            indices_range = list(range(batch_size))
        else:
            # Only applies to Commutative approach where each index utilizes 3 rows
            if cumulative_rows[last_index] < batch_size:
                num_missing_samples = (batch_size - cumulative_rows[last_index]).item()
                unused_indices = torch.arange(last_index + 1, batch_size)
                remaining_counts = row_counts[unused_indices]

                # Try to find valid samples to complete the batch
                try:
                    additional_indices = []
                    if num_missing_samples != 2:
                        valid_sample = torch.where(
                            remaining_counts == num_missing_samples
                        )[0][0].item()
                        additional_indices.append(valid_sample + last_index + 1)
                    else:
                        valid_samples = torch.where(remaining_counts == 1)[0][:2]
                        additional_indices = [
                            valid_sample.item() + last_index + 1
                            for valid_sample in valid_samples
                        ]

                    indices_range.extend(additional_indices)
                except IndexError:
                    pass

        selected_indices = sampled_indices[indices_range]
        selected_mask = initialized_mask[indices_range].flatten()

        states = self.states[selected_indices].view(-1, self.n_cells)[selected_mask]
        action_idxs = self.action_idxs[selected_indices].view(-1, 1)[selected_mask]
        rewards = self.rewards[selected_indices].view(-1, 1)[selected_mask]
        next_states = self.next_states[selected_indices].view(-1, self.n_cells)[
            selected_mask
        ]
        dones = self.dones[selected_indices].view(-1, 1)[selected_mask]
        episode_steps = self.episode_steps[selected_indices].view(-1, 1)[selected_mask]

        return (states, action_idxs, rewards, next_states, dones, episode_steps)
