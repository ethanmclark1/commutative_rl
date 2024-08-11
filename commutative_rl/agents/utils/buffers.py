import torch
import numpy as np

from typing import Union, List


torch.set_default_dtype(torch.float32)

class ReplayBuffer:
    def __init__(self, 
                 seed: int,
                 state_dims: int,
                 action_size: int,
                 buffer_size: int,
                 max_elements: int, 
                 problem_info: dict,
                 num_bridges: int
                 ) -> None:
        
        self.state_dims = state_dims
        self.max_elements = max_elements
        self.mapping = list(problem_info['mapping'].values())
        
        self.state = torch.zeros(buffer_size, state_dims)
        self.action = torch.zeros(buffer_size, action_size, dtype=torch.int64)
        self.reward = torch.zeros(buffer_size, 1)
        self.next_state = torch.zeros(buffer_size, state_dims)
        self.done = torch.zeros(buffer_size, 1, dtype=torch.bool)
        self.num_action = torch.zeros(buffer_size, 1)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)
        
        self.adapted_state = torch.zeros(buffer_size, num_bridges)
        self.adapted_next_state = torch.zeros(buffer_size, num_bridges)

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
        
        num_action = encode(num_action - 1, self.max_elements)
        
        adapted_state = adapt(state, self.mapping)
        adapted_next_state = adapt(next_state, self.mapping)
        
        self.state[self.count] = torch.as_tensor(state.flatten())
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state.flatten())
        self.done[self.count] = torch.as_tensor(done)
        self.num_action[self.count] = torch.as_tensor(num_action)
        self.adapted_state[self.count] = torch.as_tensor(adapted_state)
        self.adapted_next_state[self.count] = torch.as_tensor(adapted_next_state)
        self.is_initialized[self.count] = True

        self._increase_size()

    def sample(self, batch_size: int) -> np.ndarray:
        initialized_idxs = torch.where(self.is_initialized)[0].numpy()
        idxs = self.sample_rng.choice(initialized_idxs, size=batch_size, replace=False)
        return idxs
    
    
class RewardBuffer:
    def __init__(self, 
                 seed: int,
                 step_dims: int,
                 max_elements: int,
                 buffer_size: int,
                 action_dims: int,
                 problem_info: dict
                 ) -> None:
        
        self.action_dims = action_dims
        self.max_elements = max_elements   
        self.mapping = list(problem_info['mapping'].values())
                         
        self.transition = torch.zeros(buffer_size, step_dims) 
        self.reward = torch.zeros(buffer_size, 1)
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
            num_action: int,
            ) -> None:    
        
        action = [encode(action, self.action_dims)]
        num_action = [encode(num_action - 1, self.max_elements)]
        
        adapted_state = adapt(state, self.mapping)
        adapted_next_state = adapt(next_state, self.mapping)
            
        adapted_state = torch.as_tensor(adapted_state)
        action = torch.as_tensor(action)
        adapted_next_state = torch.as_tensor(adapted_next_state)
        num_action = torch.as_tensor(num_action)    
        
        self.transition[self.count] = torch.cat([adapted_state, action, adapted_next_state, num_action])
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
                 step_dims: int,
                 max_elements: int,
                 buffer_size: int,
                 action_dims: int,
                 problem_info: dict
                 ) -> None:
        
        super().__init__(seed, step_dims, max_elements, buffer_size, action_dims, problem_info)
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
        
        action = [encode(action, self.action_dims)]
        prev_action = [encode(prev_action, self.action_dims)]
                            
        prev_num_action = [encode(num_action - 2, self.max_elements)]
        num_action = [encode(num_action - 1, self.max_elements)]
        
        adapted_prev_state = adapt(prev_state, self.mapping)
        adapted_commutative_state = adapt(commutative_state, self.mapping)
        adapted_next_state = adapt(next_state, self.mapping)
        
        adapted_prev_state = torch.as_tensor(adapted_prev_state)
        action = torch.as_tensor(action)
        adapted_commutative_state = torch.as_tensor(adapted_commutative_state)
        prev_num_action = torch.as_tensor(prev_num_action)
        step_0 = torch.cat([adapted_prev_state, action, adapted_commutative_state, prev_num_action])
        
        prev_action = torch.as_tensor(prev_action)
        adapted_next_state = torch.as_tensor(adapted_next_state)
        num_action = torch.as_tensor(num_action)
        step_1 = torch.cat([adapted_commutative_state, prev_action, adapted_next_state, num_action])
        
        self.transition[self.count] = torch.stack((step_0, step_1))
        self.reward[self.count] = torch.as_tensor([prev_reward + reward])
        self.is_initialized[self.count] = True

        self._increase_size()
        

def encode(input_value: Union[List, torch.Tensor], dims: int) -> Union[List, torch.Tensor]:
    if isinstance(input_value, list):
        encoded = [i / (dims - 1) for i in input_value]
    else:
        encoded = input_value / (dims - 1)
    
    return encoded

def adapt(input_value: list, mapping: dict) -> list:
    adapted_input = input_value[[cell[0] for cell in mapping], [cell[1] for cell in mapping]]
    return adapted_input