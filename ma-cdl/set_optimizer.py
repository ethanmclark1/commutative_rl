import yaml
import torch
import wandb

from utils.targets_generator import generate_random_targets


class SetOptimizer:
    def __init__(self, seed: int, num_sets: int, max_action: int, action_dims: int) -> None:
        self.seed = seed        
        self.action_cost = 0.05
        self.num_sets = num_sets
        self.max_action = max_action
        self.action_dims = action_dims
        self.name = self.__class__.__name__
            
    def _init_wandb(self, problem_instance: str) -> dict:        
        wandb.init(
            project='Set Optimizer', 
            entity='ethanmclark1', 
            name=f'{self.name}',
            tags=[f'{problem_instance.capitalize()}']
            )
        
        config = wandb.config
        return config
    
    def _get_target(self, problem_instance: str, filename: str='targets.yaml') -> list:
        directory = 'ma-cdl/utils/'
        filepath = f'{directory}{filename}'
        
        while True:
            try:
                with open(filepath, 'r') as file:
                    data = yaml.safe_load(file)
                
                params = data.get('parameters', {})
                if params.get('max_action') == self.max_action and params.get('action_dims') == self.action_dims and params.get('num_sets') == self.num_sets:
                    targets = data.get('instances', {})
                    break
                else:
                    raise FileNotFoundError
                
            except FileNotFoundError:
                generate_random_targets(self.max_action, self.action_dims, self.num_sets, filepath)

        target = list(map(int, targets[problem_instance]))

        return target
        
    def _generate_start_state(self) -> tuple:        
        done = False
        num_action = 0
        state = [0] * self.max_action
        
        return state, num_action, done
    
    def _get_next_state(self, state: list, action: int) -> tuple: 
        was_tensor = True
        if not isinstance(state, torch.Tensor):
            was_tensor = False
            state = torch.as_tensor(state, dtype=torch.float) 
        
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor([action])

        next_state = state.clone()

        single_sample = len(next_state.shape) == 1
        
        next_state[next_state == 0] = self.action_dims + 1
        next_state = torch.cat([next_state, action], dim=-1)
        next_state = torch.sort(next_state, dim=-1).values
        next_state[next_state == self.action_dims + 1] = 0
        
        next_state = next_state[:-1] if single_sample else next_state[:, :-1]
        
        if not was_tensor:
            next_state = next_state.tolist()
                              
        return next_state
    
    def _calc_utility(self, state: list) -> float:    
        target = [element for element in self.target if element != 0]
        state = [element for element in map(int, state) if element != 0]        
        
        count = 0
        for element in state:
            if element in target:
                count += 1
                target.remove(element)
        
        utility = count / len(target)
        return utility
    
    def _get_reward(self, state: list, action: int, next_state: list, num_action: int) -> tuple:        
        reward = 0
        done = action == 0
        timeout = num_action == self.max_action
        
        if not done:
            util_s = self._calc_utility(state)
            util_s_prime = self._calc_utility(next_state)
            reward = util_s_prime - util_s
            reward -= self.action_cost * num_action
            
        return reward, done or timeout
            
    def _step(self, state: list, action, num_action: int) -> tuple: 
        next_state = self._get_next_state(state, action) if action != 0 else state
        reward, done = self._get_reward(state, action, next_state, num_action)
            
        return reward, next_state, done