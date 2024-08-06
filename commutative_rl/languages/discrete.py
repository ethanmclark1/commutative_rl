import copy
import wandb
import torch
import numpy as np
import more_itertools

from typing import List, Union
from commutative_rl.env import Env
from languages.utils.networks import RewardEstimator, DQN
from languages.utils.buffers import encode, ReplayBuffer, RewardBuffer, CommutativeRewardBuffer


class BasicDQN(Env):
    def __init__(self, 
                 seed: int,
                 scenario: object,
                 world: object,
                 random_state: bool,
                 train_type: str,
                 reward_type: str
                 ) -> None:
        
        super(BasicDQN, self).__init__(seed, scenario, world, random_state, train_type, reward_type)     
        self._init_hyperparams(seed)         
        self._create_candidate_set_of_lines()
        
        self.dqn = None
        self.target_dqn = None
        self.replay_buffer = None
        
        self.estimator = None
        self.reward_buffer = None
                
        self.step_dims = 2 * self.max_action + 2
        self.action_dims = len(self.candidate_lines)
        self.action_rng = np.random.default_rng(seed)
        self.hallucination_rng = np.random.default_rng(seed)
        
        self.num_action_increment = encode(1, self.max_action)

    def _init_hyperparams(self, seed: int) -> None:   
        self.seed = seed
             
        # Estimator
        self.estimator_alpha = 0.001
        self.estimator_batch_size = 128
        self.estimator_buffer_size = 7500
        
        # DQN
        self.tau = 0.008
        self.alpha = 0.003
        self.sma_window = 1
        self.granularity = 0.25
        self.min_epsilon = 0.10
        self.dqn_batch_size = 8
        self.max_powerset = 3
        self.num_episodes = 200
        self.dqn_buffer_size = 32
        self.epsilon_decay = 0.0007 if self.random_state else 0.5
        
        # Evaluation
        self.eval_freq = 1
        self.eval_window = 10
        self.eval_configs = 100
        self.eval_episodes = 1
        
    def _init_wandb(self, problem_instance: str) -> None:
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.alpha = self.alpha
        config.estimator = self.estimator
        config.eval_freq = self.eval_freq
        config.max_action = self.max_action
        config.sma_window = self.sma_window
        config.train_type = self.train_type
        config.min_epsilon = self.min_epsilon
        config.reward_type = self.reward_type
        config.action_dims = self.action_dims 
        config.eval_window = self.eval_window
        config.action_cost = self.action_cost
        config.max_powerset = self.max_powerset
        config.eval_configs = self.eval_configs
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.eval_episodes = self.eval_episodes
        config.epsilon_decay = self.epsilon_decay
        config.dqn_batch_size = self.dqn_batch_size
        config.obstacle_radius = self.obstacle_radius
        config.dqn_buffer_size = self.dqn_buffer_size
        config.util_multiplier = self.util_multiplier
        config.estimator_alpha = self.estimator_alpha
        config.failed_path_cost = self.failed_path_cost
        config.configs_to_consider = self.configs_to_consider
        config.estimator_batch_size = self.estimator_batch_size
        config.estimator_buffer_size = self.estimator_buffer_size
        config.num_large_obstacles = len(self.world.large_obstacles)
        
    def _create_candidate_set_of_lines(self) -> None:
        self.candidate_lines = []
        granularity = int(self.granularity * 100)
        
        # Terminating line
        self.candidate_lines += [(0, 0, 0)]
        
        # Using to test out simple problem (Bisect)
        # TODO: Remove after test
        self.candidate_lines += [(0.1, 0, 0.05)]
        self.candidate_lines += [(0.1, 0, 0)] 
        
        # # Vertical/Horizontal lines
        # for i in range(-100 + granularity, 100, granularity):
        #     i /= 1000
        #     self.candidate_lines += [(0.1, 0, i)] # Vertical lines
        #     self.candidate_lines += [(0, 0.1, i)] # Horizontal lines
        
        # # Diagonal lines
        # for i in range(-200 + granularity, 200, granularity):
        #     i /= 1000
        #     self.candidate_lines += [(0.1, 0.1, i)] # Left-to-Right diagonal lines
        #     self.candidate_lines += [(-0.1, 0.1, i)] # Right-to-Left diagonal lines
            
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
            elif 'Hallucinated' in self.name and done:
                non_zero_elements = sum([1 for element in state if element != 0])
                if non_zero_elements:
                    state_tensor = torch.tensor([state])
                    state, action, reward, next_state, done, num_action = self.sample_hallucinations(state_tensor)
                    
                    for i in range(state.shape[0]):
                        self.hallucinated_replay_buffer.add(state[i], action[i], reward[i], next_state[i], done[i], num_action[i])

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
            elif 'Hallucinated' in self.name and done and sum(state):
                non_zero_elements = sum([1 for element in state if element != 0])
                if non_zero_elements:
                    state_tensor = torch.tensor([state])
                    state, action, reward, next_state, done, num_action = self.sample_hallucinations(state_tensor)
                    
                    for i in range(state.shape[0]):
                        self.hallucinated_replay_buffer.add(state[i], action[i], -1, next_state[i], done[i], num_action[i])
                        self.hallucinated_reward_buffer.add(state[i], action[i], reward[i], next_state[i], num_action[i])

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
                    
    def _update_estimator(self, reward_buffer: object, losses: dict, loss_type: str) -> None:
        if self.estimator_batch_size > reward_buffer.real_size:
            return
        
        steps, rewards = reward_buffer.sample(self.estimator_batch_size)
        r_pred = self.estimator(steps)
        
        self.estimator.optim.zero_grad()
        step_loss = self.estimator.loss(r_pred, rewards)
        step_loss.backward()
        self.estimator.optim.step()
                
        losses[loss_type] += abs(step_loss.item())
    
    def _eval_policy(self, problem_instance: str) -> tuple:
        returns = []
        
        train_configs = self.configs_to_consider
        self.configs_to_consider = self.eval_configs
        
        for _ in range(self.eval_episodes):
            episode_reward = 0
            state, regions, language, num_action, done = self._generate_start_state()
        
            while not done:
                num_action += 1
                action = self._select_action(state, num_action, is_eval=True)
                reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action, num_action)
                
                line = self.candidate_lines[action]
                language.append(line)
                episode_reward += reward
                
                state = next_state
                regions = next_regions
                    
            returns.append(episode_reward)
            
        self.configs_to_consider = train_configs
        
        avg_return = np.mean(returns)
        return avg_return, language, regions
            
    def _train(self, problem_instance: str) -> tuple:    
        eval_returns = []    
        traditional_losses = []
        commutative_losses = []
        hallucinated_losses = []
        step_losses = []
        trace_losses = []
        hallucinated_step_losses = []
        
        best_return = -np.inf
        
        for episode in range(self.num_episodes):
            if episode % self.eval_freq == 0:
                eval_return, eval_language, eval_regions = self._eval_policy(problem_instance)
                eval_returns.append(eval_return)
                avg_return = np.mean(eval_returns[-self.eval_window:])
                wandb.log({'Average Return': avg_return}, step=episode)
                    
                if eval_return > best_return:
                    best_return = eval_return
                    best_language = eval_language
                    best_regions = eval_regions  
                    
            state, num_action, done = self._generate_start_state()
            
            prev_state = None
            prev_action = None
            prev_reward = None
            
            losses = {'traditional_loss': 0, 'commutative_loss': 0, 'hallucinated_loss': 0, 'step_loss': 0, 'trace_loss': 0, 'hallucinated_step_loss': 0}
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
                self._update_estimator(self.reward_buffer, losses, 'step_loss')
            
            self._learn(self.replay_buffer, losses, 'traditional_loss')
            
            traditional_losses.append(losses['traditional_loss'] / num_action)
            step_losses.append(losses['step_loss'] / num_action)
            trace_losses.append(losses['trace_loss'] / num_action)
            hallucinated_step_losses.append(losses['hallucinated_step_loss'] / num_action)
            
            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
            avg_step_loss = np.mean(step_losses[-self.sma_window:])
            avg_trace_loss = np.mean(trace_losses[-self.sma_window:])
            avg_hallucinated_step_loss = np.mean(hallucinated_losses[-self.sma_window:])
            
            wandb.log({
                "Average Traditional Loss": avg_traditional_losses,
                "Average Step Loss": avg_step_loss, 
                "Average Trace Loss": avg_trace_loss,
                "Average Hallucinated Step Loss": avg_hallucinated_step_loss}, step=episode)
            
            if losses['commutative_loss'] != 0:
                commutative_losses.append(losses['commutative_loss'] / num_action)
                avg_commutative_losses = np.mean(commutative_losses[-self.sma_window:])
                wandb.log({"Average Commutative Loss": avg_commutative_losses}, step=episode)
            elif losses['hallucinated_loss'] != 0:
                hallucinated_losses.append(losses['hallucinated_loss'] / num_action)
                avg_hallucinated_losses = np.mean(hallucinated_losses[-self.sma_window:])
                wandb.log({"Average Hallucination Loss": avg_hallucinated_losses}, step=episode)

        best_language = np.array(best_language).reshape(-1,3)
        return best_return, best_language, best_regions

    def _generate_language(self, problem_instance: str) -> np.ndarray:
        self.epsilon = 1  
        self.num_updates = 0
        self.normal_traces = 0
        self.commutative_traces = 0
        self.hallucinated_traces = 0
        
        self.dqn = DQN(self.seed, self.max_action, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        self.replay_buffer = ReplayBuffer(self.seed, self.max_action, 1, self.dqn_buffer_size, action_dims=self.action_dims)
        
        self.estimator = RewardEstimator(self.seed, self.step_dims, self.estimator_alpha)
        self.reward_buffer = RewardBuffer(self.seed, 
                                          self.step_dims,
                                          self.estimator_buffer_size,
                                          self.action_dims,
                                          self.max_action
                                          )
        
        self._init_wandb(problem_instance)
        
        best_return, best_language, best_regions = self._train(problem_instance)
                
        self._log_regions(problem_instance, 'Episode', 'Final', best_regions, best_return)
        
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
    def __init__(self, 
                 seed: int,
                 scenario: object,
                 world: object,
                 random_state: bool,
                 train_type: str,
                 reward_type: str
                 ) -> None:
        
        super(CommutativeDQN, self).__init__(seed, scenario, world, random_state, train_type, reward_type)
        
        self.commutative_replay_buffer = None
        self.commutative_reward_buffer = None
    
    def _learn(self, replay_buffer: object, losses: dict, loss_type: str) -> None:
        super()._learn(self.replay_buffer, losses, 'traditional_loss')
        super()._learn(self.commutative_replay_buffer, losses, 'commutative_loss')
            
    def _update_estimator(self, reward_buffer: object, losses: dict, loss_type: str) -> None:
        super()._update_estimator(reward_buffer, losses, 'step_loss')
        
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
    
    def _generate_language(self, problem_instance: str) -> np.ndarray:
        self.commutative_replay_buffer = ReplayBuffer(self.seed, self.max_action, 1, self.dqn_buffer_size, action_dims=self.action_dims)
        self.commutative_reward_buffer = CommutativeRewardBuffer(self.seed, 
                                                                 self.estimator_buffer_size,
                                                                 self.step_dims,
                                                                 self.action_dims,
                                                                 self.max_action
                                                                 )

        return super()._generate_language(problem_instance)
    

class HallucinatedDQN(BasicDQN):
    def __init__(self, 
                 scenario: object,
                 world: object,
                 seed: int,
                 random_state: bool,
                 train_type: str,
                 reward_type: str
                 ) -> None:
        
        super(HallucinatedDQN, self).__init__(scenario, world, seed, random_state, train_type, reward_type)
        
        self.hallucinated_replay_buffer = None
        self.hallucinated_reward_buffer = None
        
    def sample_hallucinations(self, states: Union[torch.Tensor, List]) -> tuple:        
        hallucinated_data = []
        for _state in states:
            non_zero = _state[_state != 0].numpy()
            powerset = list(more_itertools.powerset(non_zero))
            
            for subset in powerset:
                state = sorted(list(subset)) + [0] * (self.max_elements - len(subset))
                actions = np.setdiff1d(non_zero, np.array(subset))
                num_action = len(subset) + 1
                for action in actions:
                    reward, next_state, done = self._step(state, action, num_action)
                    
                    hallucinated_data.append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done,
                        'num_action': num_action,
                    })
                        
        return tuple(
            torch.tensor([h[key] for h in hallucinated_data]).unsqueeze(-1)
            if key not in ('state', 'next_state') else
            torch.tensor([h[key] for h in hallucinated_data])
            for key in ['state', 'action', 'reward', 'next_state', 'done', 'num_action']
        )
    
    def _learn(self, replay_buffer: object, losses: dict, loss_type: str) -> None:
        super()._learn(self.replay_buffer, losses, 'traditional_loss')
        super()._learn(self.hallucinated_replay_buffer, losses, 'hallucinated_loss')
        
    def _update_estimator(self, reward_buffer: object, losses: dict, loss_type: str) -> None:
        super()._update_estimator(self.reward_buffer, losses, 'step_loss')
        super()._update_estimator(self.hallucinated_reward_buffer, losses, 'hallucinated_step_loss')
        
    def generate_language(self, problem_instance: str) -> np.ndarray:
        self.hallucinated_replay_buffer = ReplayBuffer(self.seed, self.max_action, 1, self.dqn_buffer_size, action_dims=self.action_dims)
        self.hallucinated_reward_buffer = RewardBuffer(self.seed, 
                                                       self.step_dims,
                                                       self.estimator_buffer_size,
                                                       self.action_dims,
                                                       self.max_action
                                                       )
        
        return super()._generate_language(problem_instance)