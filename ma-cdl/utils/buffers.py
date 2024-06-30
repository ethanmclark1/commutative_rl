import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, 
                 seed: int,
                 state_size: int,
                 action_size: int,
                 buffer_size: int,
                 max_elements: int=None,
                 action_dims: int=None
                 ) -> None:
        
        if max_elements is None:
            max_elements = state_size
            
        self.max_elements = max_elements
        self.action_dims = action_dims
        self.action_dtype = torch.int64 if action_size == 1 else torch.float
        
        self.prev_state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.prev_action = torch.zeros(buffer_size, action_size, dtype=self.action_dtype)
        
        self.state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.action = torch.zeros(buffer_size, action_size, dtype=self.action_dtype)
        self.reward = torch.zeros(buffer_size, 1, dtype=torch.float)
        self.next_state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.done = torch.zeros(buffer_size, 1, dtype=torch.bool)
        self.num_action = torch.zeros(buffer_size, 1, dtype=torch.torch.float)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)
        
        self.has_commutative = torch.zeros(buffer_size, dtype=torch.bool)
        self.commutative_reward = torch.zeros(buffer_size, 1, dtype=torch.float)

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
            prev_state: list = None,
            prev_action: int = None,
            commutative_reward: float = None
            ) -> None:
        
        # Encode all values
        if self.action_dtype == torch.int64:
            state = encode(state, self.action_dims)
            next_state = encode(next_state, self.action_dims)
            
        num_action = encode(num_action - 1, self.max_elements)
        
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)
        self.num_action[self.count] = torch.as_tensor(num_action)
        self.is_initialized[self.count] = True
        
        if commutative_reward is not None:        
            self.prev_state[self.count] = torch.as_tensor(prev_state)
            self.prev_action[self.count] = torch.as_tensor(prev_action)
            self.commutative_reward[self.count] = torch.as_tensor(commutative_reward)
            self.has_commutative[self.count] = True

        self._increase_size()

    def sample(self, batch_size: int) -> np.ndarray:
        initialized_idxs = torch.where(self.is_initialized)[0].numpy()
        idxs = self.sample_rng.choice(initialized_idxs, size=batch_size, replace=False)
        return idxs
        

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