import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, 
                 seed: int,
                 state_size: int,
                 action_size: int,
                 buffer_size: int,
                 max_num_action: int=None,
                 max_action_index: int=None
                 ) -> None:
        
        if max_num_action is None:
            max_num_action = state_size
            
        self.max_num_action = max_num_action
        self.max_action_index = max_action_index
        
        self.action_dtype = torch.int64 if action_size == 1 else torch.float
        
        self.prev_state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.prev_action = torch.zeros(buffer_size, action_size, dtype=self.action_dtype)
        self.prev_reward = torch.zeros(buffer_size, 1, dtype=torch.float)
        self.has_previous = torch.zeros(buffer_size, dtype=torch.bool)
        
        self.state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.action = torch.zeros(buffer_size, action_size, dtype=self.action_dtype)
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
        if self.action_dtype == torch.int64:
            state = encode(state, self.max_action_index)
            num_action = encode(num_action - 1, self.max_num_action)
            next_state = encode(next_state, self.max_action_index)
        
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)
        self.num_action[self.count] = torch.as_tensor(num_action)
        self.is_initialized[self.count] = True
        
        if prev_state is not None:
            if self.action_dtype == torch.int64:
                prev_state = encode(prev_state, self.max_action_index)
            
            self.prev_state[self.count] = torch.as_tensor(prev_state)
            self.prev_action[self.count] = torch.as_tensor(prev_action)
            self.prev_reward[self.count] = torch.as_tensor(prev_reward)
            self.has_previous[self.count] = True

        self._increase_size()

    def sample(self, batch_size: int) -> torch.Tensor:
        initialized_idxs = torch.where(self.is_initialized)[0].numpy()
        idxs = self.sample_rng.choice(initialized_idxs, size=batch_size, replace=False)
        return idxs
    

# max_num_action is the maximum number of actions that can be taken in a single episode
# max_action_index is the maximum index of an action that can be taken (only applicable for DQN)
class RewardBuffer:
    def __init__(self, 
                 seed: int,
                 buffer_size: int,
                 step_dims: int,
                 max_num_action: int,
                 max_action_index: int=False
                 ) -> None:
               
        self.transition = torch.zeros((buffer_size, step_dims)) 
        self.reward = torch.zeros((buffer_size, 1))
        self.is_initialized = torch.zeros((buffer_size), dtype=torch.bool)
        
        self.action_dtype = torch.int64 if max_action_index else torch.float
        
        self.max_num_action = max_num_action
        self.max_action_index = max_action_index

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
        if self.action_dtype == torch.int64:
            state = encode(state, self.max_action_index)
            action = encode(action, self.max_action_index)
            num_action = encode(num_action - 1, self.max_num_action)
            next_state = encode(next_state, self.max_action_index)
            
            action = [action]
                
        action = torch.tensor(action).float()
        state = torch.tensor(state).float()
        next_state = torch.tensor(next_state).float() 
        reward = torch.tensor([reward]).float()
        num_action = torch.tensor([num_action]).float()       
        
        self.transition[self.count] = torch.cat([state, action, next_state, num_action])
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
    def __init__(self, 
                 seed: int,
                 buffer_size: int,
                 step_dims: int,
                 max_num_action: int,
                 max_action_index: int=None
                 ) -> None:
        
        super().__init__(seed, buffer_size, step_dims, max_num_action, max_action_index)
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
        if self.action_dtype == torch.int64:
            prev_state = encode(prev_state, self.max_action_index)
            action = encode(action, self.max_action_index)
            commutative_state = encode(commutative_state, self.max_action_index)
            prev_action = encode(prev_action, self.max_action_index)
            next_state = encode(next_state, self.max_action_index)
            
            action = [action]
            prev_action = [prev_action]
                
        prev_num_action = encode(num_action - 2, self.max_num_action)
        num_action = encode(num_action - 1, self.max_num_action)
        
        prev_state = torch.tensor(prev_state).float()
        action = torch.tensor(action).float()
        commutative_state = torch.tensor(commutative_state).float()
        prev_reward = torch.tensor([prev_reward]).float()
        prev_num_action = torch.tensor([prev_num_action]).float()
        prev_action = torch.tensor(prev_action).float()
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