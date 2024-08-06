import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam


class RewardEstimator(nn.Module):
    def __init__(self, seed: int, input_dims: int, lr: float) -> None:
        super(RewardEstimator, self).__init__()
        torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(in_features=input_dims, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=1)
        
        self.loss = nn.MSELoss()
        self.optim = Adam(self.parameters(), lr=lr)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class DQN(nn.Module):
    def __init__(self, seed: int, state_dims: int, action_dim: int, lr: float) -> None:
        super(DQN, self).__init__()
        torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_dims + 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
        self.loss = nn.MSELoss()
        self.optim = Adam(self.parameters(), lr=lr)
        
    def forward(self, state: torch.tensor, num_action: torch.tensor) -> torch.tensor:
        x = torch.cat((state, num_action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

# TODO: Update to match SAC implementation
class Actor(nn.Module):
    def __init__(self, seed: int, state_dim: int, action_dim: int, lr: float) -> None:
        super(Actor, self).__init__()
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_dim + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
        self.optim = Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.tensor, num_action: torch.tensor) -> torch.tensor:
        x = torch.cat((state, num_action), dim=-1)
        a = F.relu(self.fc1(x))
        a = F.relu(self.fc2(a))
        return torch.tanh(self.fc3(a))
    

class Critic(nn.Module):
    def __init__(self, seed: int, state_dim: int, action_dim: int, lr: float) -> None:
        super(Critic, self).__init__()
        torch.manual_seed(seed)

        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim + 1, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)
        
        self.optim = Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.tensor, action: torch.tensor, num_action: torch.tensor) -> torch.tensor:
        sa = torch.cat([state, action, num_action], dim=1)

        # Q1 architecture
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        # Q2 architecture
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    # More efficient to only compute Q1
    def Q1(self, state: torch.tensor, action: torch.tensor, num_action: torch.tensor) -> torch.tensor:
        sa = torch.cat([state, action, num_action], dim=1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1