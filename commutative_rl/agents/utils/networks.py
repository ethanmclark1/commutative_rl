import torch
import torch.nn as nn

from torch.optim import Adam


class DQN(nn.Module):
    def __init__(
        self,
        seed: int,
        state_dims: int,
        action_dims: int,
        hidden_dims: int,
        n_hidden_layers: int,
        lr: float,
        dropout: float,
    ) -> None:

        super(DQN, self).__init__()

        self.seed = torch.manual_seed(seed)

        # input layer
        layers = [
            nn.Linear(state_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        # hidden layers
        for _ in range(n_hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_dims, hidden_dims),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        # output layer
        layers.append(nn.Linear(hidden_dims, action_dims))

        self.network = nn.Sequential(*layers)

        self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        return self.network(state)
