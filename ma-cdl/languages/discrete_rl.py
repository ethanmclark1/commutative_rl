import os
import copy
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from languages.utils.cdl import CDL
from languages.utils.networks import RewardEstimator, DQN
from languages.utils.buffers import ReplayBuffer, RewardBuffer, CommutativeRewardBuffer


class BasicDQN(CDL):
    def __init__(self, scenario: object, world: object, random_state: bool, train_type: str, reward_type: str) -> None:
        super(BasicDQN, self).__init__(scenario, world, random_state)     
        self._init_hyperparams()         
        
        self.dqn = None
        self.target_dqn = None
        self.replay_buffer = None
        self.estimator = None
        self.target_estimator = None
        self.reward_buffer = None
        self.commutative_reward_buffer = None
        self.train_type = train_type
        self.reward_type = reward_type
        
        self._create_candidate_set_of_lines()
        
        self.step_dims = 2 * self.max_action + 2
        self.action_rng = np.random.default_rng(42)
        self.action_dims = len(self.candidate_lines)

    def _init_hyperparams(self) -> None:        
        # Estimator
        self.dropout_rate = 0.40
        self.estimator_tau = 0.095
        self.estimator_alpha = 0.008
        
        # DQN
        self.tau = 0.0005
        self.alpha = 0.00008
        self.batch_size = 128
        self.sma_window = 500
        self.granularity = 0.20
        self.min_epsilon = 0.10
        self.warmup_episodes = 0
        self.memory_size = 150000
        self.num_episodes = 20000
        self.epsilon_decay = 0.0005 if self.random_state else 0.0002
        
        # Evaluation
        self.eval_freq = 20
        self.eval_window = 40
        self.eval_configs = 25
        self.eval_episodes = 1
        self.eval_obstacles = 10
        
    def _init_wandb(self, problem_instance: str) -> None:
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.alpha = self.alpha
        config.estimator = self.estimator
        config.eval_freq = self.eval_freq
        config.sma_window = self.sma_window
        config.train_type = self.train_type
        config.max_action = self.max_action
        config.batch_size = self.batch_size
        config.reward_type = self.reward_type
        config.eval_window = self.eval_window
        config.min_epsilon = self.min_epsilon
        config.action_cost = self.action_cost
        config.memory_size = self.memory_size
        config.eval_configs = self.eval_configs
        config.dropout_rate = self.dropout_rate
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.estimator_tau = self.estimator_tau
        config.epsilon_decay = self.epsilon_decay
        config.eval_obstacles = self.eval_obstacles
        config.util_multiplier = self.util_multiplier
        config.estimator_alpha = self.estimator_alpha
        config.warmup_episodes = self.warmup_episodes
        config.failed_path_cost = self.failed_path_cost
        config.configs_to_consider = self.configs_to_consider
        config.num_large_obstacles = len(self.world.large_obstacles)
        
    def _create_candidate_set_of_lines(self) -> None:
        self.candidate_lines = []
        granularity = int(self.granularity * 100)
        
        # Termination Line
        self.candidate_lines += [(0, 0, 0)]
        
        # Vertical/Horizontal lines
        for i in range(-100 + granularity, 100, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0, i)] # vertical lines
            self.candidate_lines += [(0, 0.1, i)] # horizontal lines
        
        # Diagonal lines
        for i in range(-200 + granularity, 200, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0.1, i)]
            self.candidate_lines += [(-0.1, 0.1, i)]
            
    def _decrement_epsilon(self, episode: int) -> None:
        if self.reward_type == 'true' or episode > self.warmup_episodes:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)
        
    def _select_action(self, state: list, is_train: bool=True) -> tuple:
        if is_train and self.action_rng.random() < self.epsilon:
            action = self.action_rng.choice(len(self.candidate_lines))
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state) if isinstance(state, list) else state.float()
                action = self.dqn(state).argmax().item()
        
        line = self.candidate_lines[action]
        return action, line
                
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
        
        if 'Commutative' in self.name and prev_action and action != 0:   
            commutative_state = self._get_next_state(prev_state, action)
            
            self.commutative_reward_buffer.add(
                prev_state,
                action,
                prev_reward,
                commutative_state,
                num_action,
                prev_action,
                reward,
                next_state)

    def _learn(self, losses: dict) -> tuple:
        if self.replay_buffer.real_size < self.batch_size:
            return None, losses

        indices = self.replay_buffer.sample(self.batch_size)
        
        state = self.replay_buffer.state[indices]
        action = self.replay_buffer.action[indices]
        reward = self.replay_buffer.reward[indices]
        next_state = self.replay_buffer.next_state[indices]
        done = self.replay_buffer.done[indices]
        
        if self.reward_prediction_type == 'approximate':
            with torch.no_grad():
                steps = torch.cat([state, action, next_state], dim=-1)
                reward = self.target_estimator(steps).flatten()
        
        q_values = self.dqn(state)
        selected_q_values = torch.gather(q_values, 1, action).squeeze(-1)
        next_q_values = self.target_dqn(next_state)
        target_q_values = reward + ~done * torch.max(next_q_values, dim=1).values
        traditional_loss = F.mse_loss(selected_q_values, target_q_values)  
        
        self.dqn.optim.zero_grad(set_to_none=True)
        traditional_loss.backward()
        self.dqn.optim.step()
        
        self.num_updates += 1
        
        for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                    
        losses['traditional_loss'] += traditional_loss.item()
        
        return indices, losses
    
    def _update_estimator(self, losses: dict, traditional_update: bool=True) -> dict:
        if self.batch_size > self.reward_buffer.real_size:
            return losses
        
        indices = self.reward_buffer.sample(self.batch_size)
        steps = self.reward_buffer.transition[indices]
        rewards = self.reward_buffer.reward[indices].view(-1, 1)
        r_pred = self.estimator(steps)
        
        self.estimator.optim.zero_grad(set_to_none=True)
        loss = self.estimator.loss(r_pred, rewards)
        loss.backward()
        self.estimator.optim.step()
        
        if traditional_update:
            for target_param, local_param in zip(self.target_estimator.parameters(), self.estimator.parameters()):
                target_param.data.copy_(self.estimator_tau * local_param.data + (1.0 - self.estimator_tau) * target_param.data)
                
        losses['step_loss'] += loss.item()
        
        return losses
    
    def _eval_policy(self, problem_instance: str) -> tuple:
        returns = []
        
        train_configs = self.configs_to_consider
        self.configs_to_consider = self.eval_configs
        new_obstacles = self.eval_obstacles - len(self.world.large_obstacles)
        self.scenario.add_large_obstacles(self.world, new_obstacles)
        
        for _ in range(self.eval_episodes):
            episode_reward = 0
            regions, language, done = self._generate_fixed_state()
            state = sorted(list(language)) + (self.max_action - len(language)) * [0]
            num_action = len(language)
        
            while not done:
                num_action += 1
                action, line = self._select_action(state, is_train=False)
                reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action, line, num_action)
                
                language += [line]
                episode_reward += reward
                
                state = next_state
                regions = next_regions
                    
            returns.append(episode_reward)
            
        self.configs_to_consider = train_configs
        self.world.large_obstacles = self.world.large_obstacles[:-new_obstacles]
        
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
                    
            regions, language, done = self._generate_start_state()
            state = sorted(list(language)) + (self.max_action - len(language)) * [0]
            num_action = len(language)
            
            prev_state = None
            prev_action = None
            prev_reward = None
            
            losses = {'traditional_loss': 0, 'commutative_loss': 0, 'step_loss': 0, 'trace_loss': 0}
            while not done:                
                num_action += 1
                action, line = self._select_action(state)
                reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action, line, num_action)
                                
                if self.reward_type == 'approximate':
                    self._add_transition(state, action, reward, next_state, num_action, prev_state, prev_action, prev_reward)
                    
                self.replay_buffer.add(state, action, reward, next_state, done, num_action, prev_state, prev_action, prev_reward)
                       
                if 'Basic' in self.name:
                    with open(self.filename, 'a') as file:
                        file.write(f'{state}, {action}, {reward}, {next_state}, {done}, {num_action}\n')
                
                prev_state = state
                prev_action = action
                prev_reward = reward

                state = next_state
                regions = next_regions
                
            self._decrement_epsilon(episode)
            
            if self.reward_type == 'approximate':
                losses = self._update_estimator(losses)

            _, losses = self._learn(losses)
            
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
        
        prev_state = None
        prev_action = None
        prev_reward = None
                
        episode = 0
        num_step = 0
        losses = {'traditional_loss': 0, 'commutative_loss': 0, 'step_loss': 0, 'trace_loss': 0}
        for trace in history:       
            if num_step == 0 and episode % self.eval_freq == 0:
                eval_return, eval_language, eval_regions = self._eval_policy(problem_instance)
                eval_returns.append(eval_return)
                avg_return = np.mean(eval_returns[-self.eval_window:])
                wandb.log({'Average Return': avg_return}, step=episode)
                    
                if eval_return > best_return:
                    best_return = eval_return
                    best_language = eval_language
                    best_regions = eval_regions         
                        
            state, action, reward, next_state, done, num_action = trace.split(', ')
                        
            state = np.array(state, dtype=int)
            action = int(action)
            reward = float(reward)
            next_state = np.array(next_state, dtype=int)
            done = True if done == 'True' else False
            num_action = int(num_action.split('\n')[0])
            
            if num_step == 0:
                num_adaptations = np.count_nonzero(state)
            
            if self.reward_type == 'approximate':
                self._add_transition(state, action, reward, next_state, num_action, prev_state, prev_action, prev_reward)
            
            self.replay_buffer.add(state, action, reward, next_state, done, num_action, prev_state, prev_action, prev_reward)
                        
            num_step += 1
            
            prev_state = state
            prev_action = action
            prev_reward = reward
            
            if done:                
                if self.reward_type == 'approximate':
                    self._update_estimator(losses)
                    
                _, losses = self._learn(losses)
                
                traditional_losses.append(losses['traditional_loss'] / (num_action - num_adaptations))
                step_losses.append(losses['step_loss'] / (num_action - num_adaptations))
                trace_losses.append(losses['trace_loss'] / (num_action - num_adaptations))
                
                avg_traditional_td_errors = np.mean(traditional_losses[-self.sma_window:])
                avg_step_loss = np.mean(step_losses[-self.sma_window:])
                avg_trace_loss = np.mean(trace_losses[-self.sma_window:])
                
                wandb.log({
                    "Average Traditional Losses": avg_traditional_td_errors,
                    "Average Step Loss": avg_step_loss, 
                    "Average Trace Loss": avg_trace_loss}, step=episode)
                
                if losses['commutative_loss'] != 0:
                    commutative_losses.append(losses['commutative_losses'] / (num_action - num_adaptations))
                    avg_commutative_losses = np.mean(commutative_losses[-self.sma_window:])
                    wandb.log({"Average Commutative Losses": avg_commutative_losses}, step=episode)
                
                prev_state = None
                prev_action = None
                prev_reward = None
                
                episode += 1
                num_step = 0
                losses = {'traditional_loss': 0, 'commutative_loss': 0, 'step_loss': 0, 'trace_loss': 0}

        best_language = np.array(best_language).reshape(-1,3)
        return best_return, best_language, best_regions

    # Retrieve optimal set lines for a given problem instance from the training phase
    def _generate_language(self, problem_instance: str) -> np.ndarray:
        self.epsilon = 1  
        self.num_updates = 0
        
        self.filename = f'{self.output_dir}/{problem_instance}.txt'
        self.replay_buffer = ReplayBuffer(self.max_action, 1, self.memory_size)
        self.dqn = DQN(self.max_action, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        self.reward_buffer = RewardBuffer(self.batch_size, self.step_dims, self.action_dims, self.max_action)
        self.estimator = RewardEstimator(self.step_dims, self.estimator_alpha, self.dropout_rate)
        self.target_estimator = copy.deepcopy(self.estimator)
        
        self.estimator.train()
        self.target_estimator.eval()
        
        self._init_wandb(problem_instance)
        
        if self.train_type == 'online':
            best_return, best_language, best_regions = self._online_train(problem_instance)
        else:
            best_return, best_language, best_regions = self._offline_train(problem_instance)
                
        self._log_regions(problem_instance, 'Episode', 'Final', best_regions, best_return)
        wandb.log({'Language': best_language, 'Return': best_return, 'Total Updates': self.total_updates})
        wandb.finish()  
        
        return best_language
    

class CommutativeDQN(BasicDQN):
    def __init__(self, scenario: object, world: object, random_state: bool, train_type: str, reward_type: str) -> None:
        super(CommutativeDQN, self).__init__(scenario, world, random_state, train_type, reward_type)
    
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _learn(self, losses: dict) -> tuple:
        indices, losses = super()._learn(losses)
        
        if indices is None:
            return None, losses
        
        has_previous = self.replay_buffer.has_previous[indices]        
        valid_indices = torch.nonzero(has_previous, as_tuple=True)[0]
        
        prev_state = self.replay_buffer.prev_state[indices][valid_indices]
        action = self.replay_buffer.action[indices][valid_indices]
        
        commutative_state = self._get_next_state(prev_state, action)
        prev_action = self.replay_buffer.prev_action[indices][valid_indices]
        next_state = self.replay_buffer.next_state[indices][valid_indices]
        
        commutative_step = torch.cat([commutative_state, prev_action, next_state], dim=-1)
        commutative_reward = self.target_estimator(commutative_step).flatten().detach()
        
        done = self.replay_buffer.done[indices][valid_indices]            
            
        q_values = self.dqn(commutative_state)
        selected_q_values = torch.gather(q_values, 1, prev_action).squeeze(-1)
        next_q_values = self.target_dqn(next_state)
        target_q_values = commutative_reward + ~done * torch.max(next_q_values, dim=1).values
        commutative_loss = F.mse_loss(selected_q_values, target_q_values)
        
        self.dqn.optim.zero_grad(set_to_none=True)
        commutative_loss.backward()
        self.dqn.optim.step()      
        
        self.num_updates += 1
                    
        losses['commutative_loss'] += commutative_loss.item()
        
        return None, losses
    
    def _update_estimator(self, losses: dict) -> dict:
        losses = super()._update_estimator(losses, traditional_update=False)

        if self.batch_size > self.commutative_reward_buffer.real_size:
            return losses
        
        indices = self.commutative_reward_buffer.sample(self.batch_size)
        steps = self.commutative_reward_buffer.transition[indices]
        rewards = self.commutative_reward_buffer.reward[indices]
        summed_reward = torch.sum(rewards, axis=1).view(-1, 1)
        r2_pred = self.estimator(steps[:, 0])
        r3_pred = self.estimator(steps[:, 1])
                
        self.estimator.optim.zero_grad(set_to_none=True)
        loss_r2 = self.estimator.loss(r2_pred + r3_pred.detach(), summed_reward)
        loss_r3 = self.estimator.loss(r2_pred.detach() + r3_pred, summed_reward)
        combined_loss = loss_r2 + loss_r3
        combined_loss.backward()
        self.estimator.optim.step()
        
        for target_param, local_param in zip(self.target_estimator.parameters(), self.estimator.parameters()):
            target_param.data.copy_(self.estimator_tau * local_param.data + (1.0 - self.estimator_tau) * target_param.data)
            
        losses['trace_loss'] += loss_r2.item()
        
        return losses
                        
    def _generate_language(self, problem_instance: str) -> np.ndarray:
        self.commutative_reward_buffer = CommutativeRewardBuffer(self.batch_size, self.step_dims, self.action_dims, self.max_action)

        return super()._generate_language(problem_instance)