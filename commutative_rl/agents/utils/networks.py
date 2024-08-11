import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam


torch.set_default_dtype(torch.float32)

class RewardEstimator(nn.Module):
    def __init__(self, seed: int, input_dims: int, lr: float) -> None:
        super(RewardEstimator, self).__init__()
        torch.manual_seed(seed)
        
        fc_output_dims = 64
        
        self.fc1 = nn.Linear(in_features=input_dims, out_features=fc_output_dims)
        self.ln1 = nn.LayerNorm(fc_output_dims)
        self.fc2 = nn.Linear(in_features=fc_output_dims, out_features=fc_output_dims)
        self.ln2 = nn.LayerNorm(fc_output_dims)
        self.fc3 = nn.Linear(in_features=fc_output_dims, out_features=1)
        
        self.loss = nn.MSELoss()
        self.optim = Adam(self.parameters(), lr=lr)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)
    

class DQN(nn.Module):
    def __init__(self, seed: int, state_dims: int, action_dim: int, lr: float) -> None:
        super(DQN, self).__init__()
        torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_dims + 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
        self.loss = nn.MSELoss()
        self.optim = Adam(self.parameters(), lr=lr)
        
    def forward(self, state: torch.Tensor, num_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, num_action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)