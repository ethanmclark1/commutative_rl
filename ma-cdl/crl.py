import copy
import math
import wandb
import torch
import itertools
import numpy as np

from utils.networks import DQN
from set_optimizer import SetOptimizer
from utils.buffers import encode, decode, ReplayBuffer


class BasicDQN(SetOptimizer):
    def __init__(self, seed: int, num_sets: int, max_action: int, action_dims: int) -> None:
        super(BasicDQN, self).__init__(seed, num_sets, max_action, action_dims)     
        self._init_hyperparams(seed)         
        
        self.dqn = None
        self.target_dqn = None
        self.replay_buffer = None
        
        self.action_rng = np.random.default_rng(seed)
        
        self.num_action_increment = encode(1, self.max_action)

    def _init_hyperparams(self, seed: int) -> None:   
        self.seed = seed
        self.tau = 0.008
        self.alpha = 0.003
        self.eval_freq = 1
        self.sma_window = 1
        self.batch_size = 8
        self.eval_window = 10
        self.buffer_size = 32
        self.min_epsilon = 0.10
        self.num_episodes = 200
        self.epsilon_decay = 0.5
        self.max_permutations = 3
        
    def _init_wandb(self, problem_instance: str) -> None:
        config = super()._init_wandb(problem_instance)
        
        config.tau = self.tau
        config.alpha = self.alpha
        config.eval_freq = self.eval_freq
        config.max_action = self.max_action
        config.sma_window = self.sma_window
        config.batch_size = self.batch_size
        config.action_dims = self.action_dims
        config.buffer_size = self.buffer_size
        config.min_epsilon = self.min_epsilon
        config.action_dims = self.action_dims 
        config.eval_window = self.eval_window
        config.action_cost = self.action_cost
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.max_permutations = self.max_permutations
            
    def _decrement_epsilon(self) -> None:
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
    def _select_action(self, state: list, num_action: int, is_eval: bool=False) -> int:
        if is_eval or self.action_rng.random() > self.epsilon:
            state = encode(state, self.action_dims)
            num_action = encode(num_action - 1, self.max_action)
            
            state = torch.FloatTensor(state) if isinstance(state, list) else state.float()
            num_action = torch.FloatTensor([num_action])
            
            with torch.no_grad():
                action = self.dqn(state, num_action).argmax().item()
        else:
            action = self.action_rng.choice(len(self.candidate_lines))
        
        return action

    def _learn(self, losses: dict) -> np.ndarray:
        if self.replay_buffer.real_size < self.batch_size:
            return None

        indices = self.replay_buffer.sample(self.batch_size)
        
        state = self.replay_buffer.state[indices]
        action = self.replay_buffer.action[indices]
        reward = self.replay_buffer.reward[indices]      
        next_state = self.replay_buffer.next_state[indices]
        done = self.replay_buffer.done[indices]
        num_action = self.replay_buffer.num_action[indices]
        next_num_action = num_action + self.num_action_increment
        
        q_values = self.dqn(state, num_action)
        selected_q_values = torch.gather(q_values, 1, action)
        next_q_values = self.target_dqn(next_state, next_num_action)
        target_q_values = reward + ~done * torch.max(next_q_values, dim=1).values.view(-1, 1)
        
        self.num_updates += 1
        self.dqn.optim.zero_grad()
        traditional_loss = self.dqn.loss(selected_q_values, target_q_values)  
        traditional_loss.backward()
        self.dqn.optim.step()
        
        for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                    
        losses['traditional_loss'] += traditional_loss.item()
        
        return indices
    
    def _eval_policy(self) -> tuple:        
        episode_return = 0
        state, num_action, done = self._generate_start_state()
    
        while not done:
            num_action += 1
            action = self._select_action(state, num_action, is_eval=True)
            reward, next_state, done = self._step(state, action, num_action)
            
            episode_return += reward
            state = next_state
                                    
        return episode_return, state
            
    def _train(self) -> tuple:    
        eval_returns = []    
        traditional_losses = []
        commutative_losses = []
        
        best_return = -np.inf
        
        for episode in range(self.num_episodes):
            if episode % self.eval_freq == 0:
                eval_return, eval_set = self._eval_policy()
                eval_returns.append(eval_return)
                avg_return = np.mean(eval_returns[-self.eval_window:])
                wandb.log({'Average Return': avg_return}, step=episode)
                    
                if eval_return > best_return:
                    best_return = eval_return
                    best_set = eval_set
                    
            state, num_action, done = self._generate_start_state()
            
            prev_state = None
            prev_action = None
            prev_commutative_reward = None
            
            losses = {'traditional_loss': 0, 'commutative_loss': 0}
            while not done:                
                num_action += 1
                commutative_reward = None
                action = self._select_action(state, num_action)
                reward, next_state, done = self._step(state, action, num_action)
                                 
                if 'Commutative' in self.name and prev_state is not None and action != 0:
                    self.commutative_traces += 1
                    commutative_state = self._get_next_state(prev_state, action)
                    prev_commutative_reward, _ = self._get_reward(prev_state, action, commutative_state, num_action - 1)
                    commutative_reward, _ = self._get_reward(commutative_state, prev_action, next_state, num_action)
                        
                self.normal_traces += 1
                
                self.replay_buffer.add(state, action, reward, next_state, done, num_action, prev_state, prev_action, commutative_reward)          
                
                prev_state = state
                prev_action = action
                prev_reward = reward

                state = next_state
                
            self._decrement_epsilon()
            
            self._learn(losses)
            
            traditional_losses.append(losses['traditional_loss'] / (num_action - len(language)))
            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
            wandb.log({"Average Traditional Loss": avg_traditional_losses}, step=episode)
            
            if losses['commutative_loss'] != 0:
                commutative_losses.append(losses['commutative_loss'] / (num_action - len(language)))
                avg_commutative_losses = np.mean(commutative_losses[-self.sma_window:])
                wandb.log({"Average Commutative Loss": avg_commutative_losses}, step=episode)

        best_language = np.array(best_language).reshape(-1,3)
        return best_return, best_set

    def generate_target_set(self, problem_instance: str) -> np.ndarray:
        self.epsilon = 1  
        self.num_updates = 0
        self.normal_traces = 0
        self.commutative_traces = 0
        self.hallucinated_traces = 0
        
        self.target = self._get_target(problem_instance)
        self.dqn = DQN(self.seed, self.max_action, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        self.replay_buffer = ReplayBuffer(self.seed, self.max_action, 1, self.buffer_size, action_dims=self.action_dims)
                
        self._init_wandb(problem_instance)
        
        best_return, best_language = self._train()
                        
        wandb.log({
            'Language': best_language,
            'Return': best_return,
            'Total Updates': self.num_updates,
            'Normal Traces': self.normal_traces,
            'Commutative Traces': self.commutative_traces,
            'Hallucinated Traces': self.hallucinated_traces
            })
        
        wandb.finish()  
        
        return best_language
    

class CommutativeDQN(BasicDQN):
    def __init__(self, seed: int, num_sets: int, max_action, action_dims: int) -> None:
        super(CommutativeDQN, self).__init__(seed, num_sets, max_action, action_dims)
    
    def _learn(self, losses: dict) -> None:
        indices = super()._learn(losses)
        
        if indices is None:
            return 
        
        has_commutative = self.replay_buffer.has_commutative[indices]        
        commutative_indices = torch.where(has_commutative)[0]
        prev_state = self.replay_buffer.prev_state[indices][commutative_indices]
        action = self.replay_buffer.action[indices][commutative_indices]
        
        commutative_state = self._get_next_state(prev_state, action)
        commutative_state = encode(commutative_state, self.action_dims)
        prev_action = self.replay_buffer.prev_action[indices][commutative_indices]
        commutative_reward = self.replay_buffer.commutative_reward[indices][commutative_indices]
        next_state = self.replay_buffer.next_state[indices][commutative_indices]
        done = self.replay_buffer.done[indices][commutative_indices]            
        num_action = self.replay_buffer.num_action[indices][commutative_indices]
        next_num_action = num_action + self.num_action_increment
                
        q_values = self.dqn(commutative_state, num_action)
        selected_q_values = torch.gather(q_values, 1, prev_action)
        next_q_values = self.target_dqn(next_state, next_num_action)
        target_q_values = commutative_reward + ~done * torch.max(next_q_values, dim=1).values.view(-1, 1)
        
        self.num_updates += 1
        self.dqn.optim.zero_grad()
        commutative_loss = self.dqn.loss(selected_q_values, target_q_values)
        commutative_loss.backward()
        self.dqn.optim.step()      
        
        for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                    
        losses['commutative_loss'] += commutative_loss.item()
    

class HallucinatedDQN(BasicDQN):
    def __init__(self, seed: int, num_sets: int, max_action, action_dims) -> None:
        super(HallucinatedDQN, self).__init__(seed, num_sets, max_action, action_dims)
        self.max_samples = math.factorial(self.max_permutations)
    
    def _generate_permutations(self, indices) -> tuple:  
        states = decode(self.replay_buffer.state[indices], self.max_action)
        actions = self.replay_buffer.action[indices]
        dones = self.replay_buffer.done[indices]
        num_actions = self.replay_buffer.num_action[indices]
        
        permuted_states = torch.zeros((len(indices), math.factorial(self.max_action), self.max_action))
        permuted_rewards = torch.zeros((len(indices), math.factorial(self.max_action), 1))
        permuted_next_states = permuted_states.clone()
                    
        for i, (_state, action) in enumerate(zip(states, actions)):
            # Removes any zero elements from the state
            non_zero = _state[_state != 0]
            
            if non_zero.size(0) <= self.max_permutations:
                all_permutations = set(itertools.permutations(non_zero.numpy()))
            else:
                all_permutations = set()
                while len(all_permutations) < self.max_samples:
                    randperm = torch.randperm(non_zero.size(0))
                    all_permutations.add(tuple(non_zero[randperm].numpy()))

            for j, permutation in enumerate(all_permutations):
                state = list(permutation) + [0] * (self.max_action - len(permutation))
                reward, next_state, _, _ = self._step(state, action, len(permutation))
                
                permuted_states[i, j].copy_(torch.as_tensor(state))
                permuted_rewards[i, j].copy_(torch.as_tensor(reward))
                permuted_next_states[i, j].copy_(torch.as_tensor(next_state))
        
        restructured_states = torch.stack([permuted_states[:, i, :] for i in range(self.max_samples)])
        restructured_rewards = torch.stack([permuted_rewards[:, i, :] for i in range(self.max_samples)])
        restructured_next_states = torch.stack([permuted_next_states[:, i, :] for i in range(self.max_samples)])
        
        # Remove any sample that has no non-zero elements
        mask_0 = torch.any(restructured_states.sum(dim=2) != 0, dim=1)
        restructured_states = restructured_states[mask_0]
        restructured_rewards = restructured_rewards[mask_0]
        restructured_next_states = restructured_next_states[mask_0]
        
        restructured_states = encode(restructured_states, self.action_dims)
        restructured_next_states = encode(restructured_next_states, self.action_dims)
        
        # Add batch dimension if necessary
        if len(restructured_states.shape) < 3:
            restructured_states = restructured_states.unsqueeze(0)
            restructured_next_states = restructured_next_states.unsqueeze(0)
        
        return restructured_states, actions, restructured_rewards, restructured_next_states, dones, num_actions
    
    def _learn(self, losses: dict) -> None:
        indices = super()._learn(losses)
        
        if indices is None:
            return
        
        state, action, reward, next_state, done, num_action = self._generate_permutations(indices)
        next_num_action = num_action + self.num_action_increment
        
        for i in range(state.size(0)):
            q_values = self.dqn(state[i], num_action)
            selected_q_values = torch.gather(q_values, 1, action)
            next_q_values = self.target_dqn(next_state[i], next_num_action)
            target_q_values = reward[i] + ~done * torch.max(next_q_values, dim=1).values.view(-1, 1)
            
            self.num_updates += 1
            self.dqn.optim.zero_grad()
            traditional_loss = self.dqn.loss(selected_q_values, target_q_values)
            traditional_loss.backward()
            self.dqn.optim.step()
            
            for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                        
            losses['traditional_loss'] += traditional_loss.item()
            
        losses['traditional_loss'] /= state.size(0) + 1