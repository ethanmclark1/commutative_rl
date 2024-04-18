import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, state_size: int, action_size: int, buffer_size: int) -> None:
        action_dtype = torch.int64 if action_size == 1 else torch.float
        prev_state = state_size if action_dtype == torch.int64 else state_size*action_size
        
        self.prev_state = torch.zeros(buffer_size, prev_state, dtype=torch.float)
        self.prev_action = torch.zeros(buffer_size, action_size, dtype=action_dtype)
        self.prev_reward = torch.zeros(buffer_size, dtype=torch.float)
        self.prev_num_action = torch.zeros(buffer_size, dtype=torch.int64)
        
        self.state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.action = torch.zeros(buffer_size, action_size, dtype=action_dtype)
        self.reward = torch.zeros(buffer_size, dtype=torch.float)
        self.next_state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.done = torch.zeros(buffer_size, dtype=torch.bool)
        self.num_action = torch.zeros(buffer_size, dtype=torch.int64)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)
                
        self.has_previous = torch.zeros(buffer_size, dtype=torch.bool)

        self.rng = np.random.default_rng(42)
        # Managing buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        
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
                
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)
        self.num_action[self.count] = torch.as_tensor(num_action)
        self.is_initialized[self.count] = True
        
        if prev_state is not None:
            self.prev_state[self.count] = torch.as_tensor(prev_state)
            self.prev_action[self.count] = torch.as_tensor(prev_action)
            self.prev_reward[self.count] = torch.as_tensor(prev_reward)
            self.has_previous[self.count] = True

        self._increase_size()

    def sample(self, batch_size: int) -> torch.Tensor:
        initialized_idxs = torch.where(self.is_initialized == 1)[0]
        idxs = torch.multinomial(initialized_idxs.float(), batch_size, replacement=False)
        return idxs
    

# Store steps (s,a,s') -> r to estimate r hat
class RewardBuffer:
    def __init__(self, buffer_size: int, step_dims: int, action_dims: int, max_action: int) -> None:       
        self.transition = torch.zeros(buffer_size, step_dims, dtype=torch.float) 
        self.reward = torch.zeros(buffer_size, dtype=torch.float)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)
        
        self.action_dims = action_dims
        self.max_action = max_action

        self.rng = np.random.default_rng(42)
        # Managing buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        
    def _increase_size(self) -> None:
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)
        
    def encode(self, action: int, num_action: int) -> tuple:
        action_enc = torch.tensor([action / (self.action_dims - 1)]).float()
        num_action_enc = torch.tensor([(num_action - 1) / (self.max_action - 1)]).float()
        
        return action_enc, num_action_enc

    def add(self, 
            state: list,
            action: int,
            reward: float,
            next_state: list,
            num_action: int
            ) -> None:    
         
        action_enc, num_action_enc = self.encode(action, num_action)
        
        state = [i / (self.action_dims - 1) for i in state]
        next_state = [i / (self.action_dims - 1) for i in next_state]
        
        state = torch.tensor(state).float()
        next_state = torch.tensor(next_state).float()        
        
        self.transition[self.count] = torch.cat((state, action_enc, next_state, num_action_enc), dim=0)
        self.reward[self.count] = torch.tensor((reward))
        self.is_initialized[self.count] = True

        self._increase_size()

    def sample(self, batch_size: int) -> torch.Tensor:
        initialized_idxs = torch.where(self.is_initialized == 1)[0]
        idxs = torch.multinomial(initialized_idxs.float(), batch_size, replacement=False)
        return idxs
    

class CommutativeRewardBuffer(RewardBuffer):
    def __init__(self, buffer_size: int, step_dims: int, action_dims: int, max_action: int) -> None:
        super().__init__(buffer_size, step_dims, action_dims, max_action)
        self.transition = torch.zeros((buffer_size, 2, step_dims), dtype=torch.float)
        self.reward = torch.zeros(buffer_size, 2, dtype=torch.float)
        
    def add(self, 
            prev_state: list,
            action: int, 
            prev_reward: float,
            commutative_state: list,
            num_action: int,
            prev_action: int,
            reward: float,
            next_state: list
            ) -> None:
        
        prev_action_enc, prev_num_action_enc = self.encode(prev_action, num_action - 1)
        action_enc, num_action_enc = self.encode(action, num_action)
        
        prev_state = [i / (self.action_dims - 1) for i in prev_state]
        commutative_state = [i / (self.action_dims - 1) for i in commutative_state]
        next_state = [i / (self.action_dims - 1) for i in next_state]
        
        prev_state = torch.tensor(prev_state).float()
        commutative_state = torch.tensor(commutative_state).float()
        next_state = torch.tensor(next_state).float()
        
        step_0 = torch.cat((prev_state, action_enc, commutative_state, prev_num_action_enc), dim=0)
        step_1 = torch.cat((commutative_state, prev_action_enc, next_state, num_action_enc), dim=0)
        self.transition[self.count] = torch.stack((step_0, step_1))
        self.reward[self.count] = torch.tensor((prev_reward, reward))
        self.is_initialized[self.count] = True

        self._increase_size()