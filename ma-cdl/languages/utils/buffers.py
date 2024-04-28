import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, 
                 seed: int,
                 state_size: int,
                 action_size: int,
                 action_dims: int,
                 max_action: int,
                 buffer_size: int
                 ) -> None:
                
        self.action_dims = action_dims
        self.max_action = max_action
        
        action_dtype = torch.int64 if action_size == 1 else torch.float
        
        self.prev_state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.prev_action = torch.zeros(buffer_size, action_size, dtype=action_dtype)
        self.prev_reward = torch.zeros(buffer_size, 1, dtype=torch.float)
        self.has_previous = torch.zeros(buffer_size, dtype=torch.bool)
        
        self.state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.action = torch.zeros(buffer_size, action_size, dtype=action_dtype)
        self.reward = torch.zeros(buffer_size, 1, dtype=torch.float)
        self.next_state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.done = torch.zeros(buffer_size, 1, dtype=torch.bool)
        self.num_action = torch.zeros(buffer_size, 1, dtype=torch.torch.float)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)

        # Managing buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.sample_rng = np.random.default_rng(seed)
        
    def _increase_size(self) -> None:
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def add(self, 
            state: list,
            action: int,
            reward: float, 
            next_state: list,
            done: bool,
            num_action: int,
            prev_state: list,
            prev_action: int,
            prev_reward: float
            ) -> None:
        
        # Encode all values
        state = encode(state, self.action_dims)
        num_action = encode(num_action - 1, self.max_action)
        next_state = encode(next_state, self.action_dims)
        
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)
        self.num_action[self.count] = torch.as_tensor(num_action)
        self.is_initialized[self.count] = True
        
        if prev_state is not None:
            prev_state = encode(prev_state, self.action_dims)
            
            self.prev_state[self.count] = torch.as_tensor(prev_state)
            self.prev_action[self.count] = torch.as_tensor(prev_action)
            self.prev_reward[self.count] = torch.as_tensor(prev_reward)
            self.has_previous[self.count] = True

        self._increase_size()

    def sample(self, batch_size: int) -> torch.Tensor:
        initialized_idxs = torch.where(self.is_initialized)[0].numpy()
        idxs = self.sample_rng.choice(initialized_idxs, size=batch_size, replace=False)
        return idxs
    

class RewardBuffer:
    def __init__(self, seed: int, buffer_size: int, step_dims: int, action_dims: int, max_action: int) -> None:       
        self.transition = torch.zeros((buffer_size, step_dims)) 
        self.reward = torch.zeros((buffer_size, 1))
        self.is_initialized = torch.zeros((buffer_size), dtype=torch.bool)
        
        self.action_dims = action_dims
        self.max_action = max_action

        # Managing buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.sample_rng = np.random.default_rng(seed)
        
    def _increase_size(self) -> None:
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def add(self, 
            state: list,
            action: int,
            reward: float,
            next_state: list,
            num_action: int
            ) -> None:    
        
        # Encode all values
        state = encode(state, self.action_dims)
        action = encode(action, self.action_dims)
        num_action = encode(num_action - 1, self.max_action)
        next_state = encode(next_state, self.action_dims)
                
        state = torch.tensor(state).float()
        action = torch.tensor([action]).float()
        next_state = torch.tensor(next_state).float() 
        reward = torch.tensor([reward]).float()
        num_action = torch.tensor([num_action]).float()       
        
        self.transition[self.count] = torch.cat([state, action, next_state, num_action], dim=0)
        self.reward[self.count] = reward
        self.is_initialized[self.count] = True

        self._increase_size()

    def sample(self, batch_size: int) -> torch.Tensor:
        initialized_idxs = torch.where(self.is_initialized)[0].numpy()
        idxs = self.sample_rng.choice(initialized_idxs, size=batch_size, replace=False)
        steps = self.transition[idxs]
        rewards = self.reward[idxs]
        return steps, rewards
    

class CommutativeRewardBuffer(RewardBuffer):
    def __init__(self, seed: int, buffer_size: int, step_dims: int, action_dims: int, max_action: int) -> None:
        super().__init__(seed, buffer_size, step_dims, action_dims, max_action)
        self.transition = torch.zeros((buffer_size, 2, step_dims))
        self.reward = torch.zeros((buffer_size, 1))
        
    def add(self, 
            prev_state: list,
            action: int, 
            prev_reward: float,
            commutative_state: list,
            prev_action: int,
            reward: float,
            next_state: list,
            num_action: int
            ) -> None:
        
        # Encode all values
        prev_state = encode(prev_state, self.action_dims)
        action = encode(action, self.action_dims)
        prev_num_action = encode(num_action - 2, self.max_action)
        commutative_state = encode(commutative_state, self.action_dims)
        prev_action = encode(prev_action, self.action_dims)
        num_action = encode(num_action - 1, self.max_action)
        next_state = encode(next_state, self.action_dims)
                
        prev_state = torch.tensor(prev_state).float()
        action = torch.tensor([action]).float()
        commutative_state = torch.tensor(commutative_state).float()
        prev_reward = torch.tensor([prev_reward]).float()
        prev_num_action = torch.tensor([prev_num_action]).float()
        prev_action = torch.tensor([prev_action]).float()
        next_state = torch.tensor(next_state).float()
        reward = torch.tensor([reward]).float()
        num_action = torch.tensor([num_action]).float()
        
        step_0 = torch.cat([prev_state, action, commutative_state, prev_num_action])
        step_1 = torch.cat([commutative_state, prev_action, next_state, num_action])
        
        self.transition[self.count] = torch.stack((step_0, step_1))
        self.reward[self.count] = prev_reward + reward
        self.is_initialized[self.count] = True

        self._increase_size()
        

def encode(input_value, dims: int) -> list:
    if isinstance(input_value, list):
        encoded_val = [i / (dims - 1) for i in input_value]
    else:
        encoded_val = input_value / (dims - 1)
    
    return encoded_val

def decode(encoded_value, dims: int) -> list:
    return encoded_value * (dims - 1)