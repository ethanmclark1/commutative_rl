import copy
import wandb
import torch
import numpy as np
import more_itertools

from utils.networks import DQN
from set_optimizer import SetOptimizer
from utils.buffers import encode, decode, ReplayBuffer


class BasicDQN(SetOptimizer):
    def __init__(self, seed: int, num_instances: int, max_elements: int, action_dims: int) -> None:
        super(BasicDQN, self).__init__(seed, num_instances, max_elements, action_dims)     
        self._init_hyperparams(seed)         
        
        self.dqn = None
        self.target_dqn = None
        self.replay_buffer = None
        
        self.action_rng = np.random.default_rng(seed)
        self.hallucination_rng = np.random.default_rng(seed)
        
        self.num_action_increment = encode(1, self.max_elements)

    def _init_hyperparams(self, seed: int) -> None:   
        self.seed = seed
        self.tau = 0.005
        self.eval_freq = 1
        self.alpha = 0.0004
        self.sma_window = 50
        self.batch_size = 128
        self.eval_window = 50
        self.max_powerset = 5
        self.min_epsilon = 0.10
        self.buffer_size = 10000
        self.num_episodes = 2500
        self.epsilon_decay = 0.0008
        
    def _init_wandb(self, problem_instance: str) -> None:
        config = super()._init_wandb(problem_instance)
        
        config.tau = self.tau
        config.alpha = self.alpha
        config.eval_freq = self.eval_freq
        config.sma_window = self.sma_window
        config.batch_size = self.batch_size
        config.action_dims = self.action_dims
        config.buffer_size = self.buffer_size
        config.min_epsilon = self.min_epsilon
        config.action_dims = self.action_dims 
        config.eval_window = self.eval_window
        config.action_cost = self.action_cost
        config.max_elements = self.max_elements
        config.max_powerset = self.max_powerset
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
            
    def _decrement_epsilon(self) -> None:
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
    def _select_action(self, state: list, num_action: int, is_eval: bool=False) -> int:
        if is_eval or self.action_rng.random() > self.epsilon:
            state = encode(state, self.action_dims)
            num_action = encode(num_action - 1, self.max_elements)
            
            state = torch.FloatTensor(state)
            num_action = torch.FloatTensor([num_action])
            
            with torch.no_grad():
                action = self.dqn(state, num_action).argmax().item()
        else:
            action = self.action_rng.integers(self.action_dims)
        
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
            
            losses = {'traditional_loss': 0, 'commutative_loss': 0}
            while not done:                
                num_action += 1
                commutative_reward = None
                action = self._select_action(state, num_action)
                reward, next_state, done = self._step(state, action, num_action)
                                 
                if 'Commutative' in self.name and prev_state is not None and action != 0:
                    self.commutative_traces += 1
                    commutative_state = self._get_next_state(prev_state, action)
                    commutative_reward, _ = self._get_reward(commutative_state, prev_action, next_state, num_action)
                        
                self.normal_traces += 1
                
                self.replay_buffer.add(state, action, reward, next_state, done, num_action, prev_state, prev_action, commutative_reward)          
                
                prev_state = state
                prev_action = action

                state = next_state
                
            self._decrement_epsilon()
            
            self._learn(losses)
            
            traditional_losses.append(losses['traditional_loss'] / num_action)
            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
            wandb.log({"Average Traditional Loss": avg_traditional_losses}, step=episode)
            
            if losses['commutative_loss'] != 0:
                commutative_losses.append(losses['commutative_loss'] / num_action)
                avg_commutative_losses = np.mean(commutative_losses[-self.sma_window:])
                wandb.log({"Average Commutative Loss": avg_commutative_losses}, step=episode)

        best_set = list(map(int, best_set))
        return best_return, best_set

    def generate_target_set(self, problem_instance: str) -> np.ndarray:
        self.epsilon = 1  
        self.num_updates = 0
        self.normal_traces = 0
        self.commutative_traces = 0
        self.hallucinated_traces = 0
        
        self.target = self._get_target(problem_instance)
        self.dqn = DQN(self.seed, self.max_elements, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        self.replay_buffer = ReplayBuffer(self.seed, self.max_elements, 1, self.buffer_size, action_dims=self.action_dims)
                
        self._init_wandb(problem_instance)
        
        best_return, best_set = self._train()
        
        found_set = tuple(best_set) == tuple(self.target)
        
        wandb.log({
            'Best Set': best_set,
            'Return': best_return,
            'Target Set': self.target,
            'Total Updates': self.num_updates,
            'Normal Traces': self.normal_traces,
            'Commutative Traces': self.commutative_traces,
            'Hallucinated Traces': self.hallucinated_traces,
            'Found Set': found_set,
            })
        
        wandb.finish()  
        
        return best_set
    
    
class CommutativeDQN(BasicDQN):
    def __init__(self, seed: int, num_instances: int, max_elements, action_dims: int) -> None:
        super(CommutativeDQN, self).__init__(seed, num_instances, max_elements, action_dims)
    
    def _learn(self, losses: dict) -> None:
        super()._learn(losses)
        
        if self.replay_buffer.commutative_real_size < self.batch_size:
            return None
        
        indices = self.replay_buffer.commutative_sample(self.batch_size)
        
        prev_state = self.replay_buffer.prev_state[indices]
        action = self.replay_buffer.action[indices]
        
        commutative_state = self._get_next_state(prev_state, action)
        commutative_state = encode(commutative_state, self.action_dims)
        prev_action = self.replay_buffer.prev_action[indices]
        commutative_reward = self.replay_buffer.commutative_reward[indices]
        next_state = self.replay_buffer.next_state[indices]
        done = self.replay_buffer.done[indices]
        num_action = self.replay_buffer.num_action[indices]
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
    def __init__(self, seed: int, num_instances: int, max_elements, action_dims) -> None:
        super(HallucinatedDQN, self).__init__(seed, num_instances, max_elements, action_dims)
        max_hallucinations = 2 ** self.max_powerset
        self.max_hallucinations_per_batch = max_hallucinations * self.max_elements
    
    def _sample_hallucinations(self, indices) -> tuple:  
        states = self.replay_buffer.state[indices]
        states = decode(states, self.action_dims).to(torch.int32)
                
        hallucinated_states = torch.zeros((len(indices), self.max_hallucinations_per_batch, self.max_elements))
        hallucinated_actions = torch.zeros((len(indices), self.max_hallucinations_per_batch, 1))
        hallucinated_rewards = torch.zeros((len(indices), self.max_hallucinations_per_batch, 1))
        hallucinated_next_states = torch.zeros((len(indices), self.max_hallucinations_per_batch, self.max_elements))
        hallucinated_dones = torch.zeros((len(indices), self.max_hallucinations_per_batch, 1))
        hallucinated_num_actions = torch.zeros((len(indices), self.max_hallucinations_per_batch, 1))
        
        # Each state is associated with its own batch, it has the entire powerset inside of it
        for i, _state in enumerate(states):
            non_zero = _state[_state != 0]
            powerset = list(more_itertools.powerset(non_zero.numpy()))
            
            self.hallucination_rng.shuffle(powerset)
            
            # Each set in the powerset is a hallucination of the state (state X action)
            j = 0
            for _set in powerset:
                state = list(_set)
                # Generate hallucinations for each action in the action space
                for action in non_zero:
                    if action.item() not in state:
                        if j >= self.max_hallucinations_per_batch:
                            break
                        
                        padded_state = sorted(state) + [0] * (self.max_elements - len(state))
                        reward, next_state, done = self._step(padded_state, action.item(), len(state) + 1)
                        
                        hallucinated_states[i][j].copy_(torch.tensor(padded_state))
                        hallucinated_actions[i][j].copy_(torch.tensor([action.item()]))
                        hallucinated_rewards[i][j].copy_(torch.tensor([reward]))
                        hallucinated_next_states[i][j].copy_(torch.tensor(next_state))
                        hallucinated_dones[i][j].copy_(torch.tensor([done]))
                        hallucinated_num_actions[i][j].copy_(torch.tensor([len(state) + 1]))
                        
                        j += 1
                        
                if j >= self.max_hallucinations_per_batch:
                    break

        restructured_states = torch.stack([hallucinated_states[:, j, :] for j in range(self.max_hallucinations_per_batch)])
        restructured_actions = torch.stack([hallucinated_actions[:, j, :] for j in range(self.max_hallucinations_per_batch)])
        restructured_rewards = torch.stack([hallucinated_rewards[:, j, :] for j in range(self.max_hallucinations_per_batch)])
        restructured_next_states = torch.stack([hallucinated_next_states[:, j, :] for j in range(self.max_hallucinations_per_batch)])
        restructured_dones = torch.stack([hallucinated_dones[:, j, :] for j in range(self.max_hallucinations_per_batch)])
        restructured_num_actions = torch.stack([hallucinated_num_actions[:, j, :] for j in range(self.max_hallucinations_per_batch)])
        
        # Remove any sample that has no non-zero elements
        mask_0 = torch.any(restructured_states.sum(dim=2) != 0, dim=1)
        restructured_states = restructured_states[mask_0]
        restructured_actions = restructured_actions[mask_0].to(torch.int64)
        restructured_rewards = restructured_rewards[mask_0]
        restructured_next_states = restructured_next_states[mask_0]
        restructured_dones = restructured_dones[mask_0].to(torch.bool)
        restructured_num_actions = restructured_num_actions[mask_0]
        
        restructured_states = encode(restructured_states, self.action_dims)
        restructured_next_states = encode(restructured_next_states, self.action_dims)
        
        return restructured_states, restructured_actions, restructured_rewards, restructured_next_states, restructured_dones, restructured_num_actions
    
    def _learn(self, losses: dict) -> None:
        if self.replay_buffer.real_size < self.batch_size:
            return None

        indices = self.replay_buffer.sample(self.batch_size)
        
        state, action, reward, next_state, done, num_action = self._sample_hallucinations(indices)
        next_num_action = num_action + self.num_action_increment
        
        for i in range(state.size(0)):
            q_values = self.dqn(state[i], num_action[i])
            selected_q_values = torch.gather(q_values, 1, action[i])
            next_q_values = self.target_dqn(next_state[i], next_num_action[i])
            target_q_values = reward[i] + ~done[i] * torch.max(next_q_values, dim=1).values.view(-1, 1)
            
            self.num_updates += 1
            self.dqn.optim.zero_grad()
            traditional_loss = self.dqn.loss(selected_q_values, target_q_values)
            traditional_loss.backward()
            self.dqn.optim.step()
            
            for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                        
            losses['traditional_loss'] += traditional_loss.item()
            
        losses['traditional_loss'] /= state.size(0) + 1