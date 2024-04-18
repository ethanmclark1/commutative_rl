import copy
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from languages.utils.cdl import CDL
from languages.utils.networks import RewardEstimator, Actor, Critic
from languages.utils.buffers import ReplayBuffer, RewardBuffer, CommutativeRewardBuffer

torch.manual_seed(42)


class BasicTD3(CDL):
    def __init__(self, scenario: object, world: object, random_state: bool, train_type: str, reward_type: str) -> None:
        super(BasicTD3, self).__init__(scenario, world, random_state)
        self._init_hyperparams()
        
        self.actor = None
        self.critic = None
        self.actor_target = None
        self.critic_target = None
        self.replay_buffer = None
        self.estimator = None
        self.target_estimator = None
        self.reward_estiamtor = None
        self.train_type = train_type
        self.reward_type = reward_type
        
        self.action_dims = 3
        self.action_rng = np.random.default_rng(42)
        self.state_dims = self.max_action * self.action_dims
        self.step_dims = 2 * self.state_dims + self.action_dims + 1
        
    def _init_hyperparams(self) -> None:        
        # Estimator
        self.dropout_rate = 0.4
        self.estimator_tau = 0.01
        self.estimator_alpha = 0.008
        self.model_save_interval = 2500
        
        # TD3
        self.tau = 0.005
        self.policy_freq = 2
        self.noise_clip = 0.5
        self.batch_size = 256
        self.sma_window = 500
        self.policy_noise = 0.2
        self.memory_size = 100000
        self.actor_alpha = 0.0003
        self.critic_alpha = 0.0003
        self.max_timesteps = 250000
        self.start_timesteps = 25000
        self.start_timesteps = 1000
        self.exploration_noise = 0.1
        
        # Evaluation
        self.eval_window = 10
        self.eval_freq = 5000
        self.eval_configs = 15
        self.eval_episodes = 5
        self.eval_obstacles = 10
        
    def _init_wandb(self, problem_instance: str) -> None:
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.estimator = self.estimator
        config.eval_freq = self.eval_freq
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.batch_size = self.batch_size
        config.train_type = self.train_type
        config.noise_clip = self.noise_clip
        config.reward_type = self.reward_type
        config.eval_window = self.eval_window
        config.memory_size = self.memory_size
        config.action_cost = self.action_cost
        config.policy_freq = self.policy_freq
        config.actor_alpha = self.actor_alpha
        config.critic_alpha = self.critic_alpha
        config.random_state = self.random_state
        config.dropout_rate = self.dropout_rate
        config.policy_noise = self.policy_noise
        config.eval_configs = self.eval_configs
        config.max_timesteps = self.max_timesteps
        config.estimator_tau = self.estimator_tau
        config.eval_episodes = self.eval_episodes
        config.eval_obstacles = self.eval_obstacles
        config.util_multiplier = self.util_multiplier
        config.start_timesteps = self.start_timesteps
        config.estimator_alpha = self.estimator_alpha
        config.exploration_noise = self.exploration_noise
        config.model_save_interval = self.model_save_interval
        config.configs_to_consider = self.configs_to_consider
        config.num_large_obstacles = len(self.world.large_obstacles)
        
    def _is_terminating_action(self, action: np.array) -> bool:
        threshold = 0.20
        return (abs(action) < threshold).all()
          
    def _select_action(self, state: list, timestep: bool=None, is_noisy: bool=True):
        if timestep and timestep < self.start_timesteps:
            action = self.action_rng.uniform(-1, 1, size=3)
        else:
            with torch.no_grad():
                state = torch.as_tensor(state, dtype=torch.float32)
                action = self.actor(state)
                
                if is_noisy:
                    action += torch.normal(mean=0, std=self.exploration_noise, size=action.size()).clamp(-1, 1)
                    
        return action
    
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
        
        if 'Commutative' in self.name and prev_action and not self._is_terminating_action(action):   
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
                        
    def _learn(self, timestep: int, losses: dict) -> tuple:
        indices = self.replay_buffer.sample(self.batch_size)
        
        state = self.replay_buffer.state[indices]
        action = self.replay_buffer.action[indices]
        reward = self.replay_buffer.reward[indices].view(-1, 1)
        next_state = self.replay_buffer.next_state[indices]
        done = self.replay_buffer.done[indices].view(-1, 1)
        
        with torch.no_grad():
            if self.reward_type == 'approximate':
                steps = torch.cat([state, action, next_state], dim=-1)
                reward = self.target_estimator(steps).detach()      
                    
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ~done * target_Q
        
        current_Q1, current_Q2 = self.critic(state, action)
        
        traditional_critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic.optim.zero_grad(set_to_none=True)
        traditional_critic_loss.backward()
        self.critic.optim.step()
        
        losses['traditional_critic_loss'] += traditional_critic_loss.item()
        
        if timestep % self.policy_freq == 0:
            traditional_actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            losses['traditional_actor_loss'] += traditional_actor_loss.item()
            
            self.actor.optim.zero_grad(set_to_none=True)
            traditional_actor_loss.backward()
            self.actor.optim.step()
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
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

        empty_action = np.array([0.] * 3)
        
        training_configs = self.configs_to_consider
        self.configs_to_consider = self.eval_configs
        new_obstacles = self.eval_obstacles - len(self.world.large_obstacles)
        self.scenario.add_large_obstacles(self.world, new_obstacles)
        
        for _ in range(self.eval_episodes):
            episode_reward = 0
            regions, language, done = self._generate_fixed_state()
            state = np.concatenate(sorted(list(language), key=np.sum) + (self.max_action - len(language)) * [empty_action])
            num_action = len(language)
            
            while not done:
                num_action += 1
                action = self._select_action(state, is_noisy=False)
                reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action=action, line=None, num_action=num_action)
                
                language += [action]
                episode_reward += reward
                
                state = next_state
                regions = next_regions
            
            returns.append(episode_reward)
        
        self.configs_to_consider = training_configs
        self.world.large_obstacles = self.world.large_obstacles[:-new_obstacles]
        
        avg_return = np.mean(returns)
        return avg_return, language
    
    def _online_train(self, problem_instance: str) -> tuple:
        eval_returns = []
        traditional_actor_losses = []
        traditional_critic_losses = []
        commutative_actor_losses = []
        commutative_critic_losses = []
        step_losses = []
        trace_losses = []
                
        empty_action = np.array([0.] * 3)
        regions, language, done = self._generate_init_state()
        state = np.concatenate(sorted(list(language), key=np.sum) + (self.max_action - len(language)) * [empty_action])
        num_action = len(language)
        
        best_return = -np.inf
        
        prev_state = None
        prev_action = None
        prev_reward = None
        
        episode = 0 
        losses = {'traditional_actor_loss': 0, 'traditional_critic_loss': 0, 'step_loss': 0, 'trace_loss': 0}
        for timestep in range(self.max_timesteps):
            if timestep % self.eval_freq == 0:
                eval_return, eval_language, eval_regions = self._eval_policy(problem_instance)
                eval_returns.append(eval_return)
                avg_return = np.mean(eval_returns[-self.eval_window:])
                wandb.log({'Average Return': avg_return}, step=episode)
                
                if eval_return > best_return:
                    best_return = eval_return
                    best_language = eval_language
                    best_regions = eval_regions
            
            # TODO: Ensure that num_action isnt affected by eval_policy
            num_action += 1
            action = self._select_action(state, timestep)    
            reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action=action, line=None, num_action=num_action)
            
            if self.reward_type == 'approximate':
                self._add_transition(state, action, reward, next_state, prev_state, prev_action, prev_reward)
            
            self.replay_buffer.add(state, action, reward, next_state, done, prev_state, prev_action, prev_reward)
            
            if 'Basic' in self.name:
                with open(self.filename, 'a') as file:
                    file.write(f'{state}, {action}, {reward}, {next_state}, {done}, {num_action}\n')
            
            prev_state = state
            prev_action = action
            prev_reward = reward
            
            state = next_state
            regions = next_regions
            
            if timestep >= self.start_timesteps:
                _, losses = self._learn(timestep, losses)
            
            if done:
                if self.reward_type == 'approximate':
                    self._update_estimator(losses)
            
                if timestep >= self.start_timesteps:
                    traditional_actor_losses.append(losses['traditional_actor_loss'] / (num_action - len(language)))
                    traditional_critic_losses.append(losses['traditional_critic_loss'] / (num_action - len(language)))
                    step_losses.append(losses['step_loss'] / (num_action - len(language)))
                    trace_losses.append(losses['trace_loss'] / (num_action - len(language)))
                    
                    avg_traditional_actor_losses = np.mean(traditional_actor_losses[-self.sma_window:])
                    avg_traditional_critic_losses = np.mean(traditional_critic_losses[-self.sma_window:])
                    avg_step_losses = np.mean(step_losses[-self.sma_window:])
                    avg_trace_losses = np.mean(trace_losses[-self.sma_window:])
                
                    wandb.log({
                        "Average Traditional Actor Loss": avg_traditional_actor_losses,
                        "Average Traditional Critic Loss": avg_traditional_critic_losses,
                        'Average Step Loss': avg_step_losses,
                        'Average Trace Loss': avg_trace_losses,
                        }, step=episode)
                    
                    if losses['commutative_critic_loss'] != 0:
                        commutative_actor_losses.append(losses['commutative_actor_loss'] / (num_action - len(language)))
                        commutative_critic_losses.append(losses['commutative_critic_loss'] / (num_action - len(language)))
                        
                        avg_commutative_critic_losses = np.mean(commutative_critic_losses[-self.sma_window:])
                        avg_commutative_actor_losses = np.mean(commutative_actor_losses[-self.sma_window:])
                        
                        wandb.log({
                            'Average Commutative Actor Loss': avg_commutative_actor_losses,
                            'Average Commutative Critic Loss': avg_commutative_critic_losses,
                            }, step=episode)
                
                regions, language, done = self._generate_init_state()
                state = np.concatenate(sorted(list(language), key=np.sum) + (self.max_action - len(language)) * [empty_action])
                num_action = len(language)   
                
                prev_state = None
                prev_action = None
                prev_reward = None   
                
                episode += 1
                losses = {'traditional_actor_loss': 0, 'traditional_critic_loss': 0, 'step_loss': 0, 'trace_loss': 0}
                
        best_language = np.array(best_language).reshape(-1,3)
        return best_return, best_language, best_regions
    
    def _offline_train(self, problem_instance: str) -> tuple:   
        with open(self.filename, 'r') as file:
            history = file.readlines()
            
        eval_returns = []
        traditional_actor_losses = []
        traditional_critic_losses = []
        commutative_actor_losses = []
        commutative_critic_losses = []
        step_losses = []
        trace_losses = []
                
        best_return = -np.inf
        
        prev_state = None
        prev_action = None
        prev_reward = None
        
        episode = 0
        timestep = 0 
        num_step = 0
        losses = {'traditional_actor_loss': 0, 'traditional_critic_loss': 0, 'step_loss': 0, 'trace_loss': 0}
        for trace in history:
            if num_step == 0 and timestep % self.eval_freq == 0:
                eval_return, eval_language, eval_regions = self._eval_policy(problem_instance)
                eval_returns.append(eval_return)
                avg_return = np.mean(eval_returns[-self.eval_window:])
                wandb.log({'Average Return': avg_return}, step=episode)
                    
                if eval_return > best_return:
                    best_return = eval_return
                    best_language = eval_language
                    best_regions = eval_regions     
        
            state, action, reward, next_state, done, num_action = trace.split(',')
            
            # TODO: Ensure that this is working as expected
            state = np.array(state)
            action = np.array(action)
            reward = np.array(reward)
            next_state = np.array(next_state)
            done = np.array(done)
            num_action = int(num_action)
            
            if num_step == 0:
                language = np.count_nonzero(state)
                
            if self.reward_type == 'approximate':
                self._add_transition(state, action, reward, next_state, prev_state, prev_action, prev_reward)
            
            self.replay_buffer.add(state, action, reward, next_state, done, num_action, prev_state, prev_action, prev_reward)

            timestep += 1
            num_step += 1
            
            prev_state = state
            prev_action = action
            prev_reward = reward
                        
            if done:
                if self.reward_type == 'approximate':
                    self._update_estimator(losses)
            
                if timestep >= self.start_timesteps:
                    traditional_actor_losses.append(losses['traditional_actor_loss'] / (num_action - len(language)))
                    traditional_critic_losses.append(losses['traditional_critic_loss'] / (num_action - len(language)))
                    step_losses.append(losses['step_loss'] / (num_action - len(language)))
                    trace_losses.append(losses['trace_loss'] / (num_action - len(language)))
                    
                    avg_traditional_actor_losses = np.mean(traditional_actor_losses[-self.sma_window:])
                    avg_traditional_critic_losses = np.mean(traditional_critic_losses[-self.sma_window:])
                    avg_step_losses = np.mean(step_losses[-self.sma_window:])
                    avg_trace_losses = np.mean(trace_losses[-self.sma_window:])
                
                    wandb.log({
                        "Average Traditional Actor Loss": avg_traditional_actor_losses,
                        "Average Traditional Critic Loss": avg_traditional_critic_losses,
                        'Average Step Loss': avg_step_losses,
                        'Average Trace Loss': avg_trace_losses,
                        }, step=episode)
                    
                    if losses['commutative_critic_loss'] != 0:
                        commutative_actor_losses.append(losses['commutative_actor_loss'] / (num_action - len(language)))
                        commutative_critic_losses.append(losses['commutative_critic_loss'] / (num_action - len(language)))
                        
                        avg_commutative_critic_losses = np.mean(commutative_critic_losses[-self.sma_window:])
                        avg_commutative_actor_losses = np.mean(commutative_actor_losses[-self.sma_window:])
                        
                        wandb.log({
                            'Average Commutative Actor Loss': avg_commutative_actor_losses,
                            'Average Commutative Critic Loss': avg_commutative_critic_losses,
                            }, step=episode)
                
                    prev_state = None
                    prev_action = None
                    prev_reward = None   
                    
                    episode += 1
                    num_step = 0
                    losses = {'traditional_actor_loss': 0, 'traditional_critic_loss': 0, 'step_loss': 0, 'trace_loss': 0}
    
        best_language = np.array(best_language).reshape(-1,3)
        return best_return, best_language, best_regions

    def _generate_language(self, problem_instance: str) -> np.array:
        self.num_updates = 0
        
        self.filename = f'{self.output_dir}/{problem_instance}.txt'
        self.replay_buffer = ReplayBuffer(self.state_dims, self.action_dims, self.memory_size)
        self.actor = Actor(self.state_dims, self.action_dims, self.actor_alpha)
        self.critic = Critic(self.state_dims, self.action_dims, self.critic_alpha)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
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
        wandb.log({"Language": best_language, 'Final Reward': best_return})
        wandb.finish()  
        
        return best_language
    

class CommutativeTD3(BasicTD3):
    def __init__(self, scenario: object, world: object, random_state: bool, train_type: str, reward_type: str) -> None:
        super(CommutativeTD3, self).__init__(scenario, world, random_state, train_type, reward_type)
        
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
        
    def _learn(self, timestep: int, losses: dict) -> tuple:
        indices, losses = super()._learn(timestep, losses)
        
        if indices is None:
            return None, losses
        
        has_previous = self.replay_buffer.has_previous[indices]
        valid_indices = torch.nonzero(has_previous, as_tuple=True)[0]
        
        prev_state = self.replay_buffer.prev_state[indices][valid_indices]
        action = self.replay_buffer.action[indices][valid_indices]
        
        commutative_state = self._get_next_state(prev_state, action)
        prev_action = self.replay_buffer.prev_action[indices][valid_indices]
        next_state = self.replay_buffer.next_state[indices][valid_indices]
        
        done = self.replay_buffer.done[indices][valid_indices].view(-1, 1)        
        
        with torch.no_grad():        
            commutative_step = torch.cat([commutative_state, prev_action, next_state], dim=-1)
            commutative_reward = self.target_estimator(commutative_step).detach()
            
            noise = (torch.randn_like(prev_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = commutative_reward + ~done * target_Q
        
        current_Q1, current_Q2 = self.critic(commutative_state, prev_action)
        
        commutative_critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic.optim.zero_grad(set_to_none=True)
        commutative_critic_loss.backward()
        self.critic.optim.step()
        
        losses['commutative_critic_loss'] += commutative_critic_loss.item()
        
        if timestep % self.policy_freq == 0:
            commutative_actor_loss = -self.critic.Q1(commutative_state, self.actor(commutative_state)).mean()
            
            losses['commutative_actor_loss'] += commutative_actor_loss.item()
            
            self.actor.optim.zero_grad(set_to_none=True)
            commutative_actor_loss.backward()
            self.actor.optim.step()
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                        
        return None, losses
    
    def _generate_language(self, problem_instance: str) -> np.array:
        self.commutative_reward_buffer = CommutativeRewardBuffer(self.batch_size, self.step_dims, self.action_dims, self.max_action)
        
        return super()._generate_language(problem_instance)