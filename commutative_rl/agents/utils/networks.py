import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam


class DQN(nn.Module):
    def __init__(self, seed: int, state_dims: int, action_dim: int, lr: float) -> None:
        super(DQN, self).__init__()
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_dims + 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

        self.loss = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor, episode_step: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, episode_step), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
