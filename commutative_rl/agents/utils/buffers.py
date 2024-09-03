import torch
import numpy as np

from typing import Union, List


torch.set_default_dtype(torch.float32)


class ReplayBuffer:
    def __init__(
        self,
        seed: int,
        max_elements: int,
        action_size: int,
        buffer_size: int,
        action_dims: int,
    ) -> None:

        self.action_dims = action_dims
        self.max_elements = max_elements

        self.prev_state = torch.zeros(buffer_size, max_elements)
        self.prev_action_idx = torch.zeros(buffer_size, action_size, dtype=torch.int64)
        self.prev_reward = torch.zeros(buffer_size, 1)
        self.state = torch.zeros(buffer_size, max_elements)
        self.action_idx = torch.zeros(buffer_size, action_size, dtype=torch.int64)
        self.reward = torch.zeros(buffer_size, 1)
        self.next_state = torch.zeros(buffer_size, max_elements)
        self.done = torch.zeros(buffer_size, 1, dtype=torch.bool)
        self.num_action = torch.zeros(buffer_size, 1)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.sample_rng = np.random.default_rng(seed)

    def _increase_size(self) -> None:
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def add(
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

        prev_state = encode(prev_state, self.action_dims)
        state = encode(state, self.action_dims)
        next_state = encode(next_state, self.action_dims)
        num_action = encode(num_action - 1, self.max_elements)

        self.prev_state[self.count] = torch.as_tensor(prev_state)
        self.prev_action_idx[self.count] = torch.as_tensor(prev_action_idx)
        self.prev_reward[self.count] = torch.as_tensor(prev_reward)
        self.state[self.count] = torch.as_tensor(state)
        self.action_idx[self.count] = torch.as_tensor(action_idx)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)
        self.num_action[self.count] = torch.as_tensor(num_action)
        self.is_initialized[self.count] = True

        self._increase_size()

    def sample(self, batch_size: int) -> np.ndarray:
        initialized_idxs = torch.where(self.is_initialized)[0].numpy()
        idxs = self.sample_rng.choice(initialized_idxs, size=batch_size, replace=False)
        return idxs


class RewardBuffer:
    def __init__(
        self,
        seed: int,
        step_dims: int,
        max_elements: int,
        buffer_size: int,
        action_dims: int,
    ) -> None:

        self.action_dims = action_dims
        self.max_elements = max_elements

        self.transition = torch.zeros(buffer_size, 2, step_dims)
        self.reward = torch.zeros(buffer_size, 2)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.sample_rng = np.random.default_rng(seed)

    def _increase_size(self) -> None:
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def add(
        self,
        prev_state: list,
        prev_action_idx: int,
        prev_reward: float,
        state: list,
        action_idx: int,
        reward: float,
        next_state: list,
        num_action: int,
    ) -> None:

        prev_state = encode(prev_state, self.action_dims)
        prev_action_idx = [encode(prev_action_idx, self.action_dims)]
        state = encode(state, self.action_dims)
        prev_num_action = [encode(num_action - 2, self.max_elements)]

        prev_state = torch.as_tensor(prev_state)
        prev_action_idx = torch.as_tensor(prev_action_idx)
        state = torch.as_tensor(state)
        prev_num_action = torch.as_tensor(prev_num_action)

        step_0 = torch.cat([prev_state, prev_action_idx, state, prev_num_action])

        action_idx = [encode(action_idx, self.action_dims)]
        next_state = encode(next_state, self.action_dims)
        num_action = [encode(num_action - 1, self.max_elements)]

        action_idx = torch.as_tensor(action_idx)
        next_state = torch.as_tensor(next_state)
        num_action = torch.as_tensor(num_action)

        step_1 = torch.cat([state, action_idx, next_state, num_action])

        self.transition[self.count] = torch.stack((step_0, step_1))
        self.reward[self.count] = torch.as_tensor([prev_reward, reward])
        self.is_initialized[self.count] = True

        self._increase_size()

    def sample(self, batch_size: int) -> torch.Tensor:
        initialized_idxs = torch.where(self.is_initialized)[0].numpy()
        idxs = self.sample_rng.choice(initialized_idxs, size=batch_size, replace=False)
        return idxs


def encode(
    input_value: Union[int, List, torch.Tensor], dims: int, to_tensor: bool = False
) -> Union[int, torch.Tensor]:
    if isinstance(input_value, list):
        encoded = [i / (dims - 1) for i in input_value]
    else:
        encoded = input_value / (dims - 1)

    if to_tensor:
        encoded = torch.as_tensor(encoded).view(-1)

    return encoded
