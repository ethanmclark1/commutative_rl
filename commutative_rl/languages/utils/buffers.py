import torch
import numpy as np


torch.set_default_dtype(torch.float32)

class ReplayBuffer:
    def __init__(self, 
                 seed: int,
                 state_size: int,
                 action_size: int,
                 buffer_size: int,
                 max_action: int=None,
                 action_dims: int=None
                 ) -> None:
        
        if max_action is None:
            max_action = state_size
            
        self.max_action = max_action
        self.action_dims = action_dims
        self.action_dtype = torch.int64 if action_size == 1 else torch.float
        
        self.state = torch.zeros(buffer_size, state_size)
        self.action = torch.zeros(buffer_size, action_size, dtype=self.action_dtype)
        self.reward = torch.zeros(buffer_size, 1)
        self.next_state = torch.zeros(buffer_size, state_size)
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

    def add(self, 
            state: list,
            action: int,
            reward: float, 
            next_state: list,
            done: bool,
            num_action: int,
            ) -> None:
        
        # Encode all values
        if self.action_dtype == torch.int64:
            state = encode(state, self.action_dims)
            next_state = encode(next_state, self.action_dims)
            
        num_action = encode(num_action - 1, self.max_action)
        
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
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
    def __init__(self, 
                 seed: int,
                 buffer_size: int,
                 step_dims: int,
                 action_dims: int=None,
                 max_action: int=None,
                 ) -> None:
               
        self.transition = torch.zeros((buffer_size, step_dims)) 
        self.reward = torch.zeros((buffer_size, 1))
        self.is_initialized = torch.zeros((buffer_size), dtype=torch.bool)
        
        self.action_dtype = torch.int64 if action_dims else torch.float
        
        self.max_action = max_action
        self.action_dims = action_dims

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
            state = encode(state, self.action_dims)
            action = [encode(action, self.action_dims)]
            next_state = encode(next_state, self.action_dims)
                            
        num_action = [encode(num_action - 1, self.max_action)]
        
        state = torch.as_tensor(state)
        action = torch.as_tensor(action)
        next_state = torch.as_tensor(next_state)
        num_action = torch.as_tensor(num_action)       
        
        self.transition[self.count] = torch.cat([state, action, next_state, num_action])
        self.reward[self.count] = torch.as_tensor([reward])
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
                 action_dims: int=None,
                 max_action: int=None
                 ) -> None:
        
        super().__init__(seed, buffer_size, step_dims, action_dims, max_action)
        self.transition = torch.zeros((buffer_size, 2, step_dims))
        
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
            prev_state = encode(prev_state, self.action_dims)
            action = [encode(action, self.action_dims)]
            commutative_state = encode(commutative_state, self.action_dims)
            prev_action = [encode(prev_action, self.action_dims)]
            next_state = encode(next_state, self.action_dims)
                            
        prev_num_action = [encode(num_action - 2, self.max_action)]
        num_action = [encode(num_action - 1, self.max_action)]
        
        prev_state = torch.as_tensor(prev_state)
        action = torch.as_tensor(action)
        commutative_state = torch.as_tensor(commutative_state)
        prev_num_action = torch.as_tensor(prev_num_action)
        step_0 = torch.cat([prev_state, action, commutative_state, prev_num_action])
        
        prev_action = torch.as_tensor(prev_action)
        next_state = torch.as_tensor(next_state)
        num_action = torch.as_tensor(num_action)
        step_1 = torch.cat([commutative_state, prev_action, next_state, num_action])
        
        self.transition[self.count] = torch.stack((step_0, step_1))
        self.reward[self.count] = torch.as_tensor([prev_reward + reward])
        self.is_initialized[self.count] = True

        self._increase_size()
        

def encode(input_value, dims: int) -> list:
    if isinstance(input_value, list):
        encoded_val = [i / (dims - 1) for i in input_value]
    else:
        encoded_val = input_value / (dims - 1)
    
    return encoded_val

def decode(input_value, dims: int) -> list:
    if isinstance(input_value, list):
        decoded_val = [i * (dims - 1) for i in input_value]
    else:
        decoded_val = input_value * (dims - 1)
    
    return decoded_val