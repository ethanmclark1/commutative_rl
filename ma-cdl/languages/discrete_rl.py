import copy
import wandb
import torch
import numpy as np

from languages.utils.cdl import CDL
from languages.utils.networks import RewardEstimator, DQN
from languages.utils.buffers import encode, ReplayBuffer, RewardBuffer, CommutativeRewardBuffer


class BasicDQN(CDL):
    def __init__(self, 
                 scenario: object,
                 world: object,
                 seed: int,
                 random_state: bool,
                 train_type: str,
                 reward_type: str
                 ) -> None:
        
        super(BasicDQN, self).__init__(scenario, world, seed, random_state, train_type, reward_type)     
        self._init_hyperparams(seed)         
        self._create_candidate_set_of_lines()
        
        self.dqn = None
        self.estimator = None
        self.target_dqn = None
        self.replay_buffer = None
        self.reward_buffer = None
        self.commutative_reward_buffer = None
        
        self.step_dims = 2 * self.max_action + 2
        self.action_dims = len(self.candidate_lines)
        self.action_rng = np.random.default_rng(seed)
        
        self.num_action_increment = encode(1, self.max_action)

    def _init_hyperparams(self, seed: int) -> None:   
        self.seed = seed
             
        # Estimator
        self.estimator_alpha = 0.001
        self.estimator_batch_size = 128
        self.estimator_buffer_size = 7500
        
        # DQN
        self.tau = 0.003
        self.alpha = 0.0002
        self.sma_window = 100
        self.granularity = 0.20
        self.min_epsilon = 0.10
        self.dqn_batch_size = 128
        self.num_episodes = 100000
        self.dqn_buffer_size = 10000
        self.epsilon_decay = 0.0007 if self.random_state else 0.0004
        
        # Evaluation
        self.eval_freq = 100
        self.eval_window = 100
        self.eval_episodes = 1
        self.eval_configs = 75
        
    def _init_wandb(self, problem_instance: str) -> None:
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.alpha = self.alpha
        config.estimator = self.estimator
        config.eval_freq = self.eval_freq
        config.max_action = self.max_action
        config.sma_window = self.sma_window
        config.train_type = self.train_type
        config.reward_type = self.reward_type
        config.eval_window = self.eval_window
        config.min_epsilon = self.min_epsilon
        config.action_cost = self.action_cost
        config.eval_configs = self.eval_configs
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.dqn_batch_size = self.dqn_batch_size
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
        
        # Vertical/Horizontal lines
        for i in range(-40 + granularity, 40, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0, i)] # vertical lines
            self.candidate_lines += [(0, 0.1, i)] # horizontal lines
        
        # Diagonal lines
        for i in range(-200 + granularity, 200, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0.1, i)]
            self.candidate_lines += [(-0.1, 0.1, i)]
            
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
    
    def _simulate_reward(self,
                         problem_instance: str,
                         prev_state: list,
                         prev_action: int, 
                         action: int,
                         next_state: list,
                         num_action: int
                         ) -> float:
        
        commutative_state = self._get_next_state(prev_state, action)
        
        commutative_regions = self._get_next_regions(commutative_state)
        next_regions = self._get_next_regions(next_state)

        commutative_reward, _ = self._get_reward(problem_instance, commutative_regions, prev_action, next_regions, num_action)

        return commutative_reward
                
    def _add_transition(self, 
                        state: list,
                        action: int,
                        reward: float,
                        next_state: list,
                        num_action: int,
                        prev_state: list,
                        prev_action: int,
                        prev_reward: float
                        ) -> None:
        
        self.reward_buffer.add(state, action, reward, next_state, num_action)
        
        if 'Commutative' in self.name and prev_state is not None and action != 0:   
            commutative_state = self._get_next_state(prev_state, action)
            
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

    def _learn(self, problem_instance: str, losses: dict) -> np.ndarray:
        if self.replay_buffer.real_size < self.dqn_batch_size:
            return None

        indices = self.replay_buffer.sample(self.dqn_batch_size)
        
        state = self.replay_buffer.state[indices]
        action = self.replay_buffer.action[indices]
        next_state = self.replay_buffer.next_state[indices]
        done = self.replay_buffer.done[indices]
        num_action = self.replay_buffer.num_action[indices]
        next_num_action = num_action + self.num_action_increment
          
        if self.reward_type == 'true':
            reward = self.replay_buffer.reward[indices]      
        elif self.reward_type == 'approximate':
            action_enc = encode(action, self.action_dims)
            features = torch.cat([state, action_enc, next_state, num_action], dim=-1)
            
            with torch.no_grad():
                reward = self.estimator(features)
        
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
    
    def _eval_policy(self, problem_instance: str) -> tuple:
        returns = []
        
        train_configs = self.configs_to_consider
        self.configs_to_consider = self.eval_configs
        
        for _ in range(self.eval_episodes):
            episode_reward = 0
            state, regions, language, num_action, done = self._generate_fixed_state()
        
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
            
    def _online_train(self, problem_instance: str) -> tuple:    
        eval_returns = []    
        traditional_losses = []
        commutative_losses = []
        step_losses = []
        trace_losses = []
        
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
                    
            state, regions, language, num_action, done = self._generate_start_state()
            
            prev_state = None
            prev_action = None
            prev_reward = None
            
            losses = {'traditional_loss': 0, 'commutative_loss': 0, 'step_loss': 0, 'trace_loss': 0}
            while not done:                
                num_action += 1
                commutative_reward = None
                action = self._select_action(state, num_action)
                reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action, num_action)
                                
                if self.reward_type == 'approximate':
                    self._add_transition(state, action, reward, next_state, num_action, prev_state, prev_action, prev_reward) 
                elif 'Commutative' in self.name and self.reward_type == 'true' and prev_state is not None:
                    commutative_reward = self._simulate_reward(problem_instance, prev_state, prev_action, action, next_state, num_action)
                    
                self.replay_buffer.add(state, action, reward, next_state, done, num_action, prev_state, prev_action, commutative_reward)          
                       
                if 'Basic' in self.name and self.reward_type == 'true':
                    with open(self.filename, 'a') as file:
                        file.write(f'{np.array(state)}, {action}, {reward}, {np.array(next_state)}, {done}, {num_action}, {np.array(prev_state)}, {prev_action}, {prev_reward}\n')
                
                prev_state = state
                prev_action = action
                prev_reward = reward

                state = next_state
                regions = next_regions
                
            self._decrement_epsilon()
            
            if self.reward_type == 'approximate':
                self._update_estimator(losses)

            self._learn(problem_instance, losses)
            
            traditional_losses.append(losses['traditional_loss'] / (num_action - len(language)))
            step_losses.append(losses['step_loss'] / (num_action - len(language)))
            trace_losses.append(losses['trace_loss'] / (num_action - len(language)))
            
            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
            avg_step_loss = np.mean(step_losses[-self.sma_window:])
            avg_trace_loss = np.mean(trace_losses[-self.sma_window:])
            
            wandb.log({
                "Average Traditional Loss": avg_traditional_losses,
                "Average Step Loss": avg_step_loss, 
                "Average Trace Loss": avg_trace_loss}, step=episode)
            
            if losses['commutative_loss'] != 0:
                commutative_losses.append(losses['commutative_loss'] / (num_action - len(language)))
                avg_commutative_losses = np.mean(commutative_losses[-self.sma_window:])
                wandb.log({"Average Commutative Loss": avg_commutative_losses}, step=episode)

        best_language = np.array(best_language).reshape(-1,3)
        return best_return, best_language, best_regions
    
    def _offline_train(self, problem_instance: str) -> tuple:    
        with open(self.filename, 'r') as file:   
            history = file.readlines() 
        
        eval_returns = []    
        traditional_losses = []
        commutative_losses = []
        step_losses = []
        trace_losses = []
        
        best_return = -np.inf
                
        episode = 0
        losses = {'traditional_loss': 0, 'commutative_loss': 0, 'step_loss': 0, 'trace_loss': 0}
        for trace in history:     
            state, action, reward, next_state, done, num_action, prev_state, prev_action, prev_reward = trace.split(', ')

            if prev_state == 'None' and episode % self.eval_freq == 0:
                eval_return, eval_language, eval_regions = self._eval_policy(problem_instance)
                eval_returns.append(eval_return)
                avg_return = np.mean(eval_returns[-self.eval_window:])
                wandb.log({'Average Return': avg_return}, step=episode)
                    
                if eval_return > best_return:
                    best_return = eval_return
                    best_language = eval_language
                    best_regions = eval_regions         
                                            
            state = np.array(state[1:-1].split(), dtype=int)
            action = int(action)
            reward = float(reward)
            next_state = np.array(next_state[1:-1].split(), dtype=int)
            done = True if done == 'True' else False
            num_action = int(num_action)
            
            commutative_reward = None
            
            if prev_state == 'None':
                prev_state = None
                prev_action = None
                prev_reward = None
                num_adaptations = np.count_nonzero(state)
            else:
                prev_state = np.array(prev_state[1:-1].split(), dtype=int)
                prev_action = int(prev_action)
                prev_reward = float(prev_reward.split('\n')[0])
                        
            if self.reward_type == 'approximate':
                self._add_transition(state, action, reward, next_state, num_action, prev_state, prev_action, prev_reward) 
            elif 'Commutative' in self.name and self.reward_type == 'true' and prev_state is not None:
                commutative_reward = self._simulate_reward(problem_instance, prev_state, prev_action, action, next_state, num_action)
                
            self.replay_buffer.add(state, action, reward, next_state, done, num_action, prev_state, prev_action, commutative_reward)               
            
            if done:                
                if self.reward_type == 'approximate':
                    self._update_estimator(losses)

                self._learn(problem_instance, losses)
                
                traditional_losses.append(losses['traditional_loss'] / (num_action - num_adaptations))
                step_losses.append(losses['step_loss'] / (num_action - num_adaptations))
                trace_losses.append(losses['trace_loss'] / (num_action - num_adaptations))
                
                avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
                avg_step_loss = np.mean(step_losses[-self.sma_window:])
                avg_trace_loss = np.mean(trace_losses[-self.sma_window:])
                
                wandb.log({
                    "Average Traditional Loss": avg_traditional_losses,
                    "Average Step Loss": avg_step_loss, 
                    "Average Trace Loss": avg_trace_loss}, step=episode)
                
                if losses['commutative_loss'] != 0:
                    commutative_losses.append(losses['commutative_loss'] / (num_action - num_adaptations))
                    avg_commutative_losses = np.mean(commutative_losses[-self.sma_window:])
                    wandb.log({"Average Commutative Loss": avg_commutative_losses}, step=episode)
                
                episode += 1
                losses = {'traditional_loss': 0, 'commutative_loss': 0, 'step_loss': 0, 'trace_loss': 0}

        best_language = np.array(best_language).reshape(-1,3)
        return best_return, best_language, best_regions

    def _generate_language(self, problem_instance: str) -> np.ndarray:
        self.epsilon = 1  
        self.num_updates = 0
        
        self.filename = f'{self.output_dir}/{problem_instance}.txt'
        self.dqn = DQN(self.seed, self.max_action, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        self.replay_buffer = ReplayBuffer(self.seed, self.max_action, 1, self.dqn_buffer_size, action_dims=self.action_dims)
        
        self.estimator = RewardEstimator(self.seed, self.step_dims, self.estimator_alpha)
        self.reward_buffer = RewardBuffer(self.seed, self.estimator_buffer_size, self.step_dims, self.action_dims, self.max_action)
        
        self._init_wandb(problem_instance)
        
        if self.train_type == 'online':
            best_return, best_language, best_regions = self._online_train(problem_instance)
        else:
            best_return, best_language, best_regions = self._offline_train(problem_instance)
                
        self._log_regions(problem_instance, 'Episode', 'Final', best_regions, best_return)
        
        wandb.log({
            'Language': best_language,
            'Return': best_return,
            'Total Updates': self.num_updates})
        
        wandb.finish()  
        
        return best_language
    

class CommutativeDQN(BasicDQN):
    def __init__(self, 
                 scenario: object,
                 world: object,
                 seed: int,
                 random_state: bool,
                 train_type: str,
                 reward_type: str
                 ) -> None:
        
        super(CommutativeDQN, self).__init__(scenario, world, seed, random_state, train_type, reward_type)
    
    def _learn(self, problem_instance: str, losses: dict) -> None:
        indices = super()._learn(problem_instance, losses)
        
        if indices is None:
            return 
        
        has_commutative = self.replay_buffer.has_commutative[indices]        
        commutative_indices = torch.where(has_commutative)[0]
        prev_state = self.replay_buffer.prev_state[indices][commutative_indices]
        action = self.replay_buffer.action[indices][commutative_indices]
        
        commutative_state = self._get_next_state(prev_state, action)
        commutative_state = encode(commutative_state, self.action_dims)
        prev_action = self.replay_buffer.prev_action[indices][commutative_indices]
        next_state = self.replay_buffer.next_state[indices][commutative_indices]
        done = self.replay_buffer.done[indices][commutative_indices]            
        num_action = self.replay_buffer.num_action[indices][commutative_indices]
        next_num_action = num_action + self.num_action_increment
        
        if self.reward_type == 'true':
            commutative_reward = self.replay_buffer.commutative_reward[indices][commutative_indices]
        elif self.reward_type == 'approximate':
            prev_action_enc = encode(prev_action, self.action_dims)
            commutative_features = torch.cat([commutative_state, prev_action_enc, next_state, num_action], dim=-1)
            
            with torch.no_grad():
                commutative_reward = self.estimator(commutative_features)
                    
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
    
    def _generate_language(self, problem_instance: str) -> np.ndarray:
        self.commutative_reward_buffer = CommutativeRewardBuffer(self.seed, self.estimator_buffer_size, self.step_dims, self.action_dims, self.max_action)

        return super()._generate_language(problem_instance)