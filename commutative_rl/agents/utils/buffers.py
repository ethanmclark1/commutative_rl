import torch


class ReplayBuffer:
    def __init__(
        self,
        seed: int,
        batch_size: int,
        buffer_size: int,
        device: torch.device,
    ) -> None:

        self.batch_size = batch_size
        self.device = device

        self.states = torch.zeros(
            buffer_size, 1, dtype=torch.float32, device=self.device
        )
        self.action_idxs = torch.zeros(
            buffer_size, 1, dtype=torch.int64, device=self.device
        )
        self.rewards = torch.zeros(
            buffer_size, 1, dtype=torch.float32, device=self.device
        )
        self.next_states = torch.zeros(
            buffer_size, 1, dtype=torch.float32, device=self.device
        )
        self.dones = torch.zeros(buffer_size, 1, dtype=torch.bool, device=self.device)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.sample_rng = torch.Generator(device=self.device).manual_seed(seed)

    def _increase_size(self) -> None:
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.real_size + 1, self.size)

    def add(
        self,
        state: float,
        action_idx: int,
        reward: float,
        next_state: float,
        done: bool,
    ) -> None:

        self.states[self.count] = torch.as_tensor(state, device=self.device)
        self.action_idxs[self.count] = torch.as_tensor(action_idx, device=self.device)
        self.rewards[self.count] = torch.as_tensor(reward, device=self.device)
        self.next_states[self.count] = torch.as_tensor(next_state, device=self.device)
        self.dones[self.count] = torch.as_tensor(done, device=self.device)

        self._increase_size()

    def sample(self) -> tuple:
        indices = torch.randint(
            0,
            self.real_size,
            (self.batch_size,),
            generator=self.sample_rng,
            device=self.device,
        )

        states = self.states[indices]
        action_idxs = self.action_idxs[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        return (states, action_idxs, rewards, next_states, dones)
