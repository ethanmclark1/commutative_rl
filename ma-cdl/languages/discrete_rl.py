import copy
import wandb
import torch
import numpy as np

from languages.utils.cdl import CDL
from languages.utils.networks import RewardEstimator, DQN
from languages.utils.buffers import encode, decode, ReplayBuffer, RewardBuffer, CommutativeRewardBuffer


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
        
        self.step_dims = 2 * self.max_num_action + 2
        self.max_action_index = len(self.candidate_lines)
        self.action_rng = np.random.default_rng(seed)
        self.num_action_increment = encode(1, self.max_num_action)

    def _init_hyperparams(self, seed: int) -> None:   
        self.seed = seed
        self.batch_size = 128
        self.buffer_size = 50000
             
        # Estimator
        self.estimator_alpha = 0.0008
        
        # DQN
        self.tau = 0.005
        self.alpha = 0.0008
        self.sma_window = 250
        self.granularity = 0.20
        self.min_epsilon = 0.10
        self.num_episodes = 15000
        self.epsilon_decay = 0.0008 if self.random_state else 0.0003
        
        # Evaluation
        self.eval_freq = 100
        self.eval_window = 100
        self.eval_configs = 25
        self.eval_episodes = 1
        self.eval_obstacles = 12
        
    def _init_wandb(self, problem_instance: str) -> None:
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.alpha = self.alpha
        config.estimator = self.estimator
        config.eval_freq = self.eval_freq
        config.sma_window = self.sma_window
        config.train_type = self.train_type
        config.batch_size = self.batch_size
        config.reward_type = self.reward_type
        config.eval_window = self.eval_window
        config.min_epsilon = self.min_epsilon
        config.action_cost = self.action_cost
        config.buffer_size = self.buffer_size
        config.eval_configs = self.eval_configs
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.max_num_action = self.max_num_action
        config.eval_obstacles = self.eval_obstacles
        config.util_multiplier = self.util_multiplier
        config.estimator_alpha = self.estimator_alpha
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
            
    def _decrement_epsilon(self) -> None:
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
    def _select_action(self, state: list, num_action: int, is_eval: bool=False) -> tuple:
        if is_eval or self.action_rng.random() > self.epsilon:
            with torch.no_grad():
                state = encode(state, self.max_action_index)
                num_action = encode(num_action - 1, self.max_num_action)
                num_action = torch.FloatTensor([num_action])
                
                if isinstance(state, list):
                    state = torch.FloatTensor(state)
                else:
                    state = state.float()
                                
                action = self.dqn(state, num_action).argmax().item()
        else:
            action = self.action_rng.choice(len(self.candidate_lines))
        
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

    def _learn(self, losses: dict) -> None:
        if self.replay_buffer.real_size < self.batch_size:
            return

        indices = self.replay_buffer.sample(self.batch_size)
        
        state = self.replay_buffer.state[indices]
        action = self.replay_buffer.action[indices]
        reward = self.replay_buffer.reward[indices]
        next_state = self.replay_buffer.next_state[indices]
        done = self.replay_buffer.done[indices]
        num_action = self.replay_buffer.num_action[indices]
                
        if self.reward_type == 'approximate':
            action_enc = encode(action, self.max_action_index)
            features = torch.cat([state, action_enc, next_state, num_action], dim=-1)
            
            with torch.no_grad():
                reward = self.estimator(features)
        
        q_values = self.dqn(state, num_action)
        selected_q_values = torch.gather(q_values, 1, action)
        next_q_values = self.target_dqn(next_state, num_action + self.num_action_increment)
        target_q_values = reward + ~done * torch.max(next_q_values, dim=1).values.view(-1, 1)
        
        self.num_updates += 1
        self.dqn.optim.zero_grad()
        traditional_loss = self.dqn.loss(selected_q_values, target_q_values)  
        traditional_loss.backward()
        self.dqn.optim.step()
        
        for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                    
        losses['traditional_loss'] += traditional_loss.item()
            
    def _update_estimator(self, losses: dict) -> None:
        if self.batch_size > self.reward_buffer.real_size:
            return
        
        steps, rewards = self.reward_buffer.sample(self.batch_size)
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
        new_obstacles = self.eval_obstacles - len(self.world.large_obstacles)
        self.scenario.add_large_obstacles(self.world, new_obstacles)
        
        for _ in range(self.eval_episodes):
            episode_reward = 0
            regions, language, done = self._generate_fixed_state()
            state = sorted(list(language)) + (self.max_num_action - len(language)) * [0]
            num_action = len(language)
        
            while not done:
                num_action += 1
                action, line = self._select_action(state, num_action, is_eval=True)
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
            state = sorted(list(language)) + (self.max_num_action - len(language)) * [0]
            num_action = len(language)
            
            prev_state = None
            prev_action = None
            prev_reward = None
            
            losses = {'traditional_loss': 0, 'commutative_loss': 0, 'step_loss': 0, 'trace_loss': 0}
            while not done:                
                num_action += 1
                action, line = self._select_action(state, num_action)
                reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action, line, num_action)
                                
                self.replay_buffer.add(state, action, reward, next_state, done, num_action, prev_state, prev_action, prev_reward)
                
                if self.reward_type == 'approximate':
                    self._add_transition(state, action, reward, next_state, num_action, prev_state, prev_action, prev_reward)    
                       
                if 'Basic' in self.name:
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

            self._learn(losses)
            
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
            
            self.replay_buffer.add(state, action, reward, next_state, done, num_action, prev_state, prev_action, prev_reward)
            
            if done:                
                if self.reward_type == 'approximate':
                    self._update_estimator(losses)
                    
                self._learn(losses)
                
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
                    commutative_losses.append(losses['commutative_loss'] / (num_action - num_adaptations))
                    avg_commutative_losses = np.mean(commutative_losses[-self.sma_window:])
                    wandb.log({"Average Commutative Losses": avg_commutative_losses}, step=episode)
                
                episode += 1
                losses = {'traditional_loss': 0, 'commutative_loss': 0, 'step_loss': 0, 'trace_loss': 0}

        best_language = np.array(best_language).reshape(-1,3)
        return best_return, best_language, best_regions

    # Retrieve optimal set lines for a given problem instance from the training phase
    def _generate_language(self, problem_instance: str) -> np.ndarray:
        self.epsilon = 1  
        self.num_updates = 0
        
        self.filename = f'{self.output_dir}/{problem_instance}.txt'
        self.dqn = DQN(self.seed, self.max_num_action, self.max_action_index, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        self.replay_buffer = ReplayBuffer(
            self.seed, 
            state_size=self.max_num_action,
            action_size=1, 
            buffer_size=self.buffer_size,
            max_action_index=self.max_action_index
            )
        
        self.estimator = RewardEstimator(self.seed, self.step_dims, self.estimator_alpha)
        self.reward_buffer = RewardBuffer(self.seed, self.buffer_size, self.step_dims, self.max_num_action, self.max_action_index)
        
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
    
    def _learn(self, losses: dict) -> None:
        super()._learn(losses)
        
        if self.replay_buffer.real_size < self.batch_size:
            return
        
        indices = self.replay_buffer.sample(self.batch_size)
        has_previous = self.replay_buffer.has_previous[indices]        
        valid_indices = torch.nonzero(has_previous, as_tuple=True)[0]
        
        prev_state = self.replay_buffer.prev_state[indices][valid_indices]
        action = self.replay_buffer.action[indices][valid_indices]
        
        prev_state_decoded = decode(prev_state, self.max_action_index)
        commutative_state_decoded = self._get_next_state(prev_state_decoded, action)
        commutative_state = encode(commutative_state_decoded, self.max_action_index)
        
        prev_action = self.replay_buffer.prev_action[indices][valid_indices]
        next_state = self.replay_buffer.next_state[indices][valid_indices]
        num_action = self.replay_buffer.num_action[indices][valid_indices]
        
        prev_action_enc = encode(prev_action, self.max_action_index)
        commutative_step = torch.cat([commutative_state, prev_action_enc, next_state, num_action], dim=-1)
        with torch.no_grad():
            commutative_reward = self.estimator(commutative_step)
        
        done = self.replay_buffer.done[indices][valid_indices]            
            
        q_values = self.dqn(commutative_state, num_action)
        selected_q_values = torch.gather(q_values, 1, prev_action)
        next_q_values = self.target_dqn(next_state, num_action + self.num_action_increment)
        target_q_values = commutative_reward + ~done * torch.max(next_q_values, dim=1).values.view(-1, 1)
        
        self.num_updates += 1
        self.dqn.optim.zero_grad()
        commutative_loss = self.dqn.loss(selected_q_values, target_q_values)
        commutative_loss.backward()
        self.dqn.optim.step()      
        
        for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                    
        losses['commutative_loss'] += commutative_loss.item()
            
    def _update_estimator(self, losses: dict) -> dict:
        super()._update_estimator(losses)
        
        if self.batch_size > self.commutative_reward_buffer.real_size:
            return
        
        steps, rewards = self.commutative_reward_buffer.sample(self.batch_size)
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
        self.commutative_reward_buffer = CommutativeRewardBuffer(self.seed, self.batch_size, self.step_dims, self.max_num_action, self.max_action_index)

        return super()._generate_language(problem_instance)