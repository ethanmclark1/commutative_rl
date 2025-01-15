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
        activation_fn: int,
        n_hidden_layers: int,
        # the following args are None when testing for commutative preservation
        loss_fn: str = None,
        lr: float = None,
        layer_norm: bool = False,
    ) -> None:

        super(DQN, self).__init__()

        self.seed = torch.manual_seed(seed)

        # input layer
        layers = [
            nn.Linear(state_dims, hidden_dims),
            nn.LayerNorm(hidden_dims) if layer_norm else nn.Identity(),
            nn.ReLU() if activation_fn == "ReLU" else nn.ELU(),
        ]

        # hidden layers
        for _ in range(n_hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_dims, hidden_dims),
                    nn.LayerNorm(hidden_dims) if layer_norm else nn.Identity(),
                    nn.ReLU() if activation_fn == "ReLU" else nn.ELU(),
                ]
            )

        # output layer
        layers.append(nn.Linear(hidden_dims, action_dims))

        self.network = nn.Sequential(*layers)

        if loss_fn is not None:
            self.loss_fn = nn.SmoothL1Loss() if loss_fn == "Huber" else nn.MSELoss()
        if lr is not None:
            self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        return self.network(state)
