import copy
import wandb
import torch
import numpy as np
import more_itertools

from collections import Counter
from set_optimizer import SetOptimizer
from utils.networks import DQN, RewardEstimator
from utils.buffers import encode, decode, adapt, ReplayBuffer, RewardBuffer, CommutativeRewardBuffer


class BasicDQN(SetOptimizer):
    def __init__(self, 
                 seed: int, 
                 num_instances: int,
                 max_elements: int,
                 action_dims: int,
                 reward_type: str,
                 reward_noise: float
                 ) -> None:
        
        super(BasicDQN, self).__init__(seed, num_instances, max_elements, action_dims, reward_type, reward_noise)     
        self._init_hyperparams(seed)         
        
        self.target = None
        
        self.dqn = None
        self.target_dqn = None
        self.replay_buffer = None
        
        self.estimator = None
        self.reward_buffer = None
        
        # [adapted state, action, adapted next state, num action]
        self.step_dims = 4
        self.action_rng = np.random.default_rng(seed)
        self.hallucination_rng = np.random.default_rng(seed)
        
        self.num_action_increment = encode(1, self.max_elements)

    def _init_hyperparams(self, seed: int) -> None:   
        self.seed = seed
        
        # Estimator
        self.estimator_alpha = 0.0005
        self.estimator_batch_size = 128
        self.estimator_buffer_size = 100000
        
        # DQN
        self.tau = 0.008
        self.alpha = 0.0008
        self.sma_window = 150
        self.max_powerset = 7
        self.min_epsilon = 0.10
        self.num_episodes = 25000
        self.dqn_batch_size = 128
        self.epsilon_decay = 0.0004
        self.dqn_buffer_size = 100000
        
        # Evaluation
        self.eval_freq = 1
        self.eval_window = 150
        
    def _init_wandb(self, problem_instance: str) -> None:
        config = super()._init_wandb(problem_instance)
        
        config.tau = self.tau
        config.alpha = self.alpha
        config.estimator = self.estimator
        config.eval_freq = self.eval_freq
        config.sma_window = self.sma_window
        config.action_dims = self.action_dims
        config.min_epsilon = self.min_epsilon
        config.action_dims = self.action_dims 
        config.eval_window = self.eval_window
        config.action_cost = self.action_cost
        config.reward_noise = self.reward_noise
        config.max_elements = self.max_elements
        config.max_powerset = self.max_powerset
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.dqn_batch_size = self.dqn_batch_size
        config.dqn_buffer_size = self.dqn_buffer_size
        config.estimator_alpha = self.estimator_alpha
        config.estimator_batch_size = self.estimator_batch_size
        config.estimator_buffer_size = self.estimator_buffer_size
            
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
    
    def _add_to_buffers(self, 
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
        
        self.normal_traces += 1
        self.replay_buffer.add(state, action, reward, next_state, done, num_action)     
        
        if self.reward_type == 'true':
            if 'Commutative' in self.name and prev_state is not None and action != 0:
                commutative_state = self._get_next_state(prev_state, action)
                
                if num_action == 2:
                    self.commutative_traces += 1
                    prev_commutative_reward, _ = self._get_reward(prev_state, action, commutative_state, num_action - 1)
                    self.commutative_replay_buffer.add(prev_state, action, prev_commutative_reward, commutative_state, False, num_action - 1)
                
                self.commutative_traces += 1
                commutative_reward, _, _ = self._step(commutative_state, prev_action, num_action)
                self.commutative_replay_buffer.add(commutative_state, prev_action, commutative_reward, next_state, done, num_action)
        elif self.reward_type == 'approximate':
            self.reward_buffer.add(state, action, reward, next_state, num_action)
            if 'Commutative' in self.name and prev_state is not None and action != 0:
                commutative_state = self._get_next_state(prev_state, action)
                
                if num_action == 2:    
                    self.commutative_traces += 1
                    self.commutative_replay_buffer.add(prev_state, action, -1, commutative_state, False, num_action - 1)
                
                self.commutative_traces += 1
                self.commutative_replay_buffer.add(commutative_state, prev_action, -1, next_state, done, num_action)
                
                self.commutative_reward_buffer.add(
                    prev_state,
                    action,
                    prev_reward,
                    commutative_state,
                    prev_action,
                    reward,
                    next_state,
                    num_action
                    )

    def _learn(self, replay_buffer: object, losses: dict, loss_type: str) -> np.ndarray:
        if replay_buffer.real_size < self.dqn_batch_size:
            return None

        indices = replay_buffer.sample(self.dqn_batch_size)
        
        state = replay_buffer.state[indices]
        action = replay_buffer.action[indices]
        reward = replay_buffer.reward[indices]      
        next_state = replay_buffer.next_state[indices]
        done = replay_buffer.done[indices]
        num_action = replay_buffer.num_action[indices]
        next_num_action = num_action + self.num_action_increment
        
        adapted_state = replay_buffer.adapted_state[indices]
        adapted_next_state = replay_buffer.adapted_next_state[indices]
          
        if self.reward_type == 'approximate':
            action_enc = encode(action, self.action_dims)
            features = torch.cat([adapted_state, action_enc, adapted_next_state, num_action], dim=-1)
            
            with torch.no_grad():
                reward = self.estimator(features)
        
        q_values = self.dqn(state, num_action)
        selected_q_values = torch.gather(q_values, 1, action)
        next_q_values = self.target_dqn(next_state, next_num_action)
        target_q_values = reward + ~done * torch.max(next_q_values, dim=1).values.view(-1, 1)
        
        self.num_updates += 1
        self.dqn.optim.zero_grad()
        loss = self.dqn.loss(selected_q_values, target_q_values)  
        loss.backward()
        self.dqn.optim.step()
        
        for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                    
        losses[loss_type] += loss.item()
        
        return indices
    
    def _update_estimator(self, losses: dict) -> None:
        if self.estimator_batch_size > self.reward_buffer.real_size:
            return
        
        steps, rewards = self.reward_buffer.sample(self.estimator_batch_size)
        r_pred = self.estimator(steps)
        
        self.estimator.optim.zero_grad()
        step_loss = self.estimator.loss(r_pred, rewards)
        step_loss.backward()
        self.estimator.optim.step()
                
        losses['step_loss'] += abs(step_loss.item())
            
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
        hallucination_losses = []
        step_losses = []
        trace_losses = []
        
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
            prev_reward = None
            
            losses = {'traditional_loss': 0, 'commutative_loss': 0, 'hallucination_loss': 0, 'step_loss': 0, 'trace_loss': 0,}
            while not done:                
                num_action += 1
                action = self._select_action(state, num_action)
                reward, next_state, done = self._step(state, action, num_action)
                
                self._add_to_buffers(state, action, reward, next_state, done, num_action, prev_state, prev_action, prev_reward)
                        
                prev_state = state
                prev_action = action
                prev_reward = reward

                state = next_state
                
            self._decrement_epsilon()
            
            if self.reward_type == 'approximate':
                self._update_estimator(losses)
            
            self._learn(self.replay_buffer, losses, 'traditional_loss')
            
            traditional_losses.append(losses['traditional_loss'] / num_action)
            step_losses.append(losses['step_loss'] / num_action)
            trace_losses.append(losses['trace_loss'] / num_action)
            
            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
            avg_step_loss = np.mean(step_losses[-self.sma_window:])
            avg_trace_loss = np.mean(trace_losses[-self.sma_window:])
            
            wandb.log({
                "Average Traditional Loss": avg_traditional_losses,
                "Average Step Loss": avg_step_loss, 
                "Average Trace Loss": avg_trace_loss}, step=episode)
            
            if losses['commutative_loss'] != 0:
                commutative_losses.append(losses['commutative_loss'] / num_action)
                avg_commutative_losses = np.mean(commutative_losses[-self.sma_window:])
                wandb.log({"Average Commutative Loss": avg_commutative_losses}, step=episode)
            elif losses['hallucination_loss'] != 0:
                hallucination_losses.append(losses['hallucination_loss'] / num_action)
                avg_hallucination_losses = np.mean(hallucination_losses[-self.sma_window:])
                wandb.log({"Average Hallucination Loss": avg_hallucination_losses}, step=episode)

        best_set = list(map(int, best_set))
        return best_return, best_set

    def generate_target_set(self, problem_instance: str) -> np.ndarray:
        self.epsilon = 1  
        self.num_updates = 0
        self.normal_traces = 0
        self.commutative_traces = 0
        self.hallucinated_traces = 0
        
        if self.target is None:
            self.target = self._get_target(problem_instance)
            
        self.dqn = DQN(self.seed, self.max_elements, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        self.replay_buffer = ReplayBuffer(self.seed, self.max_elements, 1, self.dqn_buffer_size, self.action_dims, self.target)
        
        self.estimator = RewardEstimator(self.seed, self.step_dims, self.estimator_alpha)
        self.reward_buffer = RewardBuffer(self.seed, 
                                          self.step_dims, 
                                          self.max_elements,
                                          self.estimator_buffer_size,
                                          self.action_dims,
                                          self.target
                                          )
                
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
    def __init__(self, 
                 seed: int,
                 num_instances: int,
                 max_elements: int,
                 action_dims: int,
                 reward_type: str,
                 reward_noise: float
                 ) -> None:
        
        super(CommutativeDQN, self).__init__(seed, num_instances, max_elements, action_dims, reward_type, reward_noise)
        
        self.commutative_reward_buffer = None
    
    def _learn(self, replay_buffer: object, losses: dict, loss_type: str) -> None:
        super()._learn(self.replay_buffer, losses, 'traditional_loss')
        super()._learn(self.commutative_replay_buffer, losses, 'commutative_loss')
        
    def _update_estimator(self, losses: dict) -> None:
        super()._update_estimator(losses)
        
        if self.estimator_batch_size > self.commutative_reward_buffer.real_size:
            return
        
        steps, rewards = self.commutative_reward_buffer.sample(self.estimator_batch_size)
        r2_pred = self.estimator(steps[:, 0])
        r3_pred = self.estimator(steps[:, 1])
                
        self.estimator.optim.zero_grad()
        loss_r2 = self.estimator.loss(r2_pred + r3_pred.detach(), rewards)
        loss_r3 = self.estimator.loss(r2_pred.detach() + r3_pred, rewards)
        trace_loss = loss_r2 + loss_r3
        trace_loss.backward()
        self.estimator.optim.step()
            
        losses['trace_loss'] += abs(trace_loss.item() / 2)
        
    def generate_target_set(self, problem_instance: str) -> np.ndarray:
        self.target = self._get_target(problem_instance)
        
        self.commutative_replay_buffer = ReplayBuffer(self.seed, self.max_elements, 1, self.dqn_buffer_size, self.action_dims, self.target)
        self.commutative_reward_buffer = CommutativeRewardBuffer(self.seed, 
                                                                 self.step_dims,
                                                                 self.max_elements,
                                                                 self.estimator_buffer_size,
                                                                 self.action_dims,
                                                                 self.target
                                                                 )

        super().generate_target_set(problem_instance)
        

class HallucinatedDQN(BasicDQN):
    def __init__(self, 
                 seed: int,
                 num_instances: int,
                 max_elements: int,
                 action_dims: int,
                 reward_type: str,
                 reward_noise: float
                 ) -> None:
        
        super(HallucinatedDQN, self).__init__(seed, num_instances, max_elements, action_dims, reward_type, reward_noise)
                
        max_hallucinations = 2 ** self.max_powerset
        self.max_hallucinations_per_batch = max_hallucinations * self.max_elements
    
    def _sample_hallucinations(self, indices) -> tuple:
        states = self.replay_buffer.state[indices]
        states = decode(states, self.action_dims)
        
        hallucinated_data = []
        for _state in states:
            non_zero = _state[_state != 0].numpy()
            powerset = list(more_itertools.powerset(non_zero))
            self.hallucination_rng.shuffle(powerset)
            
            batch_hallucinations = []
            for subset in powerset:
                state = sorted(list(subset)) + [0] * (self.max_elements - len(subset))
                actions = np.setdiff1d(non_zero, np.array(subset))
                num_action = len(subset) + 1
                for action in actions:
                    reward, next_state, done = self._step(state, action, num_action)
                        
                    adapted_state = adapt(state, self.target_counter)
                    adapted_next_state = adapt(next_state, self.target_counter)
                    
                    batch_hallucinations.append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done,
                        'num_action': num_action,
                        'adapted_state': adapted_state,
                        'adapted_next_state': adapted_next_state
                    })
                    
                    if len(batch_hallucinations) >= self.max_hallucinations_per_batch:
                        break
                
                if len(batch_hallucinations) >= self.max_hallucinations_per_batch:
                    break
            
            if batch_hallucinations:
                hallucinated_data.extend(batch_hallucinations)
                
        assert(len(hallucinated_data) > 0)
        
        state = torch.tensor([h['state'] for h in hallucinated_data])
        action = torch.tensor([h['action'] for h in hallucinated_data]).unsqueeze(-1)
        reward = torch.tensor([h['reward'] for h in hallucinated_data]).unsqueeze(-1)
        next_state = torch.tensor([h['next_state'] for h in hallucinated_data])
        done = torch.tensor([h['done'] for h in hallucinated_data]).unsqueeze(-1)
        num_action = torch.tensor([h['num_action'] for h in hallucinated_data]).unsqueeze(-1)
        adapted_state = torch.tensor([h['adapted_state'] for h in hallucinated_data]).unsqueeze(-1)
        adapted_next_state = torch.tensor([h['adapted_next_state'] for h in hallucinated_data]).unsqueeze(-1)

        state = encode(state, self.action_dims)
        next_state = encode(next_state, self.action_dims)
        num_action = encode(num_action - 1, self.max_elements)
        adapted_state = encode(adapted_state, self.target_length + 1)
        adapted_next_state = encode(adapted_next_state, self.target_length + 1)
        
        return state, action, reward, next_state, done, num_action, adapted_state, adapted_next_state
        
    def _learn(self, replay_buffer: object, losses: dict, loss_type: str='hallucinated_loss') -> None:
        super()._learn(replay_buffer, losses, 'traditional_loss')
        
        if replay_buffer.real_size < self.dqn_batch_size:
            return None

        indices = replay_buffer.sample(self.dqn_batch_size)
        
        state, action, reward, next_state, done, num_action, adapted_state, adapted_next_state = self._sample_hallucinations(indices)
        next_num_action = num_action + self.num_action_increment
        
        if self.reward_type == 'approximate':
            action_enc = encode(action, self.action_dims)
            features = torch.cat([adapted_state, action_enc, adapted_next_state, num_action], dim=-1)
            
            with torch.no_grad():
                reward = self.estimator(features)
        
        q_values = self.dqn(state, num_action)
        selected_q_values = torch.gather(q_values, 1, action)
        next_q_values = self.target_dqn(next_state, next_num_action)
        target_q_values = reward + ~done * torch.max(next_q_values, dim=1).values.view(-1, 1)
        
        self.num_updates += 1
        self.dqn.optim.zero_grad()
        loss = self.dqn.loss(selected_q_values, target_q_values)
        loss.backward()
        self.dqn.optim.step()
        
        for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                    
        losses['hallucination_loss'] += loss.item()
        
    def generate_target_set(self, problem_instance: str) -> np.ndarray:
        target = self._get_target(problem_instance)
        self.target_counter = Counter(target)
        del self.target_counter[0]
        self.target_length = len(self.target_counter)
        
        super().generate_target_set(problem_instance)