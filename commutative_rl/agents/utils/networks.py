import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam


class DuelingDQN(nn.Module):
    def __init__(
        self,
        seed: int,
        state_dims: int,
        action_dims: int,
        hidden_dims: int,
        loss_fn: str,
        lr: float,
        layer_norm: bool = False,
    ) -> None:

        super(DuelingDQN, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.feature = nn.Sequential(
            nn.Linear(state_dims, hidden_dims),
            nn.LayerNorm(hidden_dims) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ELU(),
        )

        self.value = nn.Linear(hidden_dims, 1)
        self.advantage = nn.Linear(hidden_dims, action_dims)

        self.loss_fn = nn.SmoothL1Loss() if loss_fn == "Huber" else nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.feature(state)
        value = self.value(features)
        advantage = self.advantage(features)

        return value + (advantage - advantage.mean(dim=1, keepdim=True))
