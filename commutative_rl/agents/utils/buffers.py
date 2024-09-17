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

        self.n_steps = n_steps

        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)

        self.states = torch.zeros(buffer_size, n_cells)
        self.action_idxs = torch.zeros(buffer_size, 1, dtype=torch.int64)
        self.rewards = torch.zeros(buffer_size, 1)
        self.next_states = torch.zeros(buffer_size, n_cells)
        self.dones = torch.zeros(buffer_size, 1, dtype=torch.bool)
        self.episode_steps = torch.zeros(buffer_size, 1)

        self.corresponding_indices = torch.full((buffer_size,), -1, dtype=torch.int64)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.sample_rng = torch.Generator().manual_seed(seed)

    def _increase_size(self) -> None:
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
        corresponding_index: int = None,
    ) -> None:

        episode_step = encode(episode_step, self.n_steps)

        self.is_initialized[self.count] = True

        self.states[self.count] = torch.as_tensor(state)
        self.action_idxs[self.count] = torch.as_tensor(action_idx)
        self.rewards[self.count] = torch.as_tensor(reward)
        self.next_states[self.count] = torch.as_tensor(next_state)
        self.dones[self.count] = torch.as_tensor(done)
        self.episode_steps[self.count] = torch.as_tensor(episode_step)

        if corresponding_index is not None:
            self.corresponding_indices[self.count] = torch.as_tensor(
                corresponding_index
            )

        self._increase_size()

    def sample(self, batch_size: int) -> torch.Tensor:
        valid_indices = torch.nonzero(self.is_initialized).squeeze()
        num_valid = valid_indices.size(0)
        random_indices = torch.randperm(num_valid, generator=self.sample_rng)[
            :batch_size
        ]
        return valid_indices[random_indices]
