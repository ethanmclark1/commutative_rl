import copy
import torch
import wandb
import numpy as np

from env import Env
from .networks import DQN, RewardEstimator
from .buffers import encode, ReplayBuffer, RewardBuffer


class Parent(Env):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        min_dist_bounds: int,
        action_dims: int,
        negative_actions: bool,
        duplicate_actions: bool,
        reward_type: str,
        reward_noise: float,
    ) -> None:

        super(Parent, self).__init__(
            seed,
            num_instances,
            min_dist_bounds,
            action_dims,
            negative_actions,
            duplicate_actions,
            reward_type,
            reward_noise,
        )
        self._init_hyperparams(seed)

        self.problem = None

        self.dqn = None
        self.target_dqn = None
        self.replay_buffer = None

        self.estimator = None
        self.reward_buffer = None

        self.step_dims = 2 * self.max_elements + 2
        self.action_rng = np.random.default_rng(seed)
        self.hallucination_rng = np.random.default_rng(seed)

        self.num_action_increment = encode(1, self.max_elements)

    def _init_hyperparams(self, seed: int) -> None:
        self.seed = seed

        # Estimator
        self.estimator_alpha = 0.002
        self.estimator_batch_size = 128
        self.estimator_buffer_size = 500
        self.estimator_hallucinated_buffer_size = 100000

        # DQN
        self.tau = 0.001
        self.alpha = 0.0004
        self.batch_size = 128
        self.max_powerset = 10
        self.buffer_size = 500
        self.min_epsilon = 0.10
        self.num_episodes = 25000
        self.epsilon_decay = 0.002
        self.hallucinated_buffer_size = 100000
        self.sma_window = 15 if self.reward_noise == 0 else 250

        # Evaluation
        self.eval_freq = 1
        self.eval_window = 15 if self.reward_noise == 0 else 250

    def _init_wandb(self, problem_instance: str) -> None:
        config = super()._init_wandb(problem_instance)

        config.tau = self.tau
        config.alpha = self.alpha
        config.ub_sum = self.ub_sum
        config.lb_sum = self.lb_sum
        config.actions = self.actions
        config.estimator = self.estimator
        config.eval_freq = self.eval_freq
        config.sma_window = self.sma_window
        config.target_sum = self.target_sum
        config.batch_size = self.batch_size
        config.buffer_size = self.buffer_size
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
        config.min_dist_bounds = self.min_dist_bounds
        config.estimator_alpha = self.estimator_alpha
        config.negative_actions = self.negative_actions
        config.duplicate_actions = self.duplicate_actions
        config.estimator_batch_size = self.estimator_batch_size
        config.estimator_buffer_size = self.estimator_buffer_size
        config.hallucinated_buffer_size = self.hallucinated_buffer_size
        config.estimator_hallucinated_buffer_size = (
            self.estimator_hallucinated_buffer_size
        )

    def _decrement_epsilon(self) -> None:
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)

    def _select_action(
        self, state: list, num_action: int, is_eval: bool = False
    ) -> int:
        if is_eval or self.action_rng.random() > self.epsilon:
            state = encode(state, self.action_dims, to_tensor=True)
            num_action = encode(num_action - 1, self.max_elements, to_tensor=True)

            with torch.no_grad():
                action_idx = self.dqn(state, num_action).argmax().item()
        else:
            action_idx = self.action_rng.integers(self.action_dims)

        return action_idx

    def _add_to_buffers(
        self,
        prev_state: list,
        prev_action_idx: int,
        prev_reward: float,
        state: list,
        action_idx: int,
        reward: float,
        next_state: list,
        done: bool,
        num_action: int,
    ) -> None:

        self.normal_traces += 1

        if prev_state is not None:
            self.replay_buffer.add(
                prev_state,
                prev_action_idx,
                prev_reward,
                state,
                action_idx,
                reward,
                next_state,
                done,
                num_action,
            )

            if self.reward_type == "approximate":
                self.reward_buffer.add(
                    prev_state,
                    prev_action_idx,
                    prev_reward,
                    state,
                    action_idx,
                    reward,
                    next_state,
                    num_action,
                )

    def _update_estimator(
        self,
        losses: dict,
        reward_buffer: object = None,
        loss_type: str = None,
        indices: tuple = None,
    ) -> None:
        if (
            reward_buffer is None
            or loss_type is None
            or self.estimator_batch_size > reward_buffer.real_size
        ):
            return None

        if indices is None:
            indices = reward_buffer.sample(self.estimator_batch_size)

        transitions = reward_buffer.transition[indices]
        rewards = reward_buffer.reward[indices]

        prev_r_pred = self.estimator(transitions[:, 0])
        r_pred = self.estimator(transitions[:, 1])

        loss_prev_to_curr = self.estimator.loss(prev_r_pred, rewards[:, 0].view(-1, 1))
        loss_curr_to_next = self.estimator.loss(r_pred, rewards[:, 1].view(-1, 1))

        loss = loss_prev_to_curr + loss_curr_to_next
        loss.backward()

        losses[loss_type] += abs(loss.item())

        return indices

    def _learn(
        self,
        losses: dict,
        replay_buffer: object = None,
        loss_type: str = None,
        indices: tuple = None,
    ) -> np.ndarray:
        if (
            replay_buffer is None
            or loss_type is None
            or self.batch_size > replay_buffer.real_size
        ):
            return None

        if indices is None:
            indices = replay_buffer.sample(self.batch_size)

        prev_state = replay_buffer.prev_state[indices]
        prev_action_idx = replay_buffer.prev_action_idx[indices]
        prev_reward = replay_buffer.prev_reward[indices]
        state = replay_buffer.state[indices]
        action_idx = replay_buffer.action_idx[indices]
        reward = replay_buffer.reward[indices]
        next_state = replay_buffer.next_state[indices]
        done = replay_buffer.done[indices]

        num_action = replay_buffer.num_action[indices]
        prev_num_action = num_action - self.num_action_increment
        next_num_action = num_action + self.num_action_increment

        if self.reward_type == "approximate":
            prev_action_enc = encode(prev_action_idx, self.action_dims)
            action_enc = encode(action_idx, self.action_dims)
            prev_transition = torch.cat(
                [prev_state, prev_action_enc, state, prev_num_action], dim=-1
            )
            transition = torch.cat([state, action_enc, next_state, num_action], dim=-1)

            with torch.no_grad():
                prev_reward = self.estimator(prev_transition)
                reward = self.estimator(transition)

        prev_q_values = self.dqn(prev_state, prev_num_action)
        prev_selected_q_values = torch.gather(prev_q_values, 1, prev_action_idx)

        q_values = self.dqn(state, num_action)
        selected_q_values = torch.gather(q_values, 1, action_idx)

        with torch.no_grad():
            prev_next_q_values = self.target_dqn(state, num_action)
            next_q_values = self.target_dqn(next_state, next_num_action)

        prev_target_q_values = prev_reward + ~done * torch.max(
            prev_next_q_values, dim=1
        ).values.view(-1, 1)
        target_q_values = reward + ~done * torch.max(next_q_values, dim=1).values.view(
            -1, 1
        )

        self.num_updates += 1
        self.dqn.optim.zero_grad()

        loss_prev_to_curr = self.dqn.loss(prev_selected_q_values, prev_target_q_values)
        loss_curr_to_next = self.dqn.loss(selected_q_values, target_q_values)

        loss = loss_prev_to_curr + loss_curr_to_next
        loss.backward()

        losses[loss_type] += loss.item()

        return indices

    def _eval_policy(self) -> tuple:
        episode_return = 0
        state, num_action, done = self._generate_start_state()

        while not done:
            num_action += 1
            action_idx = self._select_action(state, num_action, is_eval=True)
            reward, next_state, done = self._step(state, action_idx, num_action)

            episode_return += reward
            state = next_state

        return episode_return, state

    def _train(self) -> tuple:
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
                eval_return, eval_set = self._eval_policy()
                eval_returns.append(eval_return)
                avg_return = np.mean(eval_returns[-self.eval_window :])
                wandb.log({"Average Return": avg_return}, step=episode)

                if eval_return > best_return:
                    best_return = eval_return
                    best_set = eval_set

            state, num_action, done = self._generate_start_state()

            prev_state = None
            prev_action_idx = None
            prev_reward = None

            losses = {
                "traditional_loss": 0,
                "commutative_loss": 0,
                "hallucinated_loss": 0,
                "step_loss": 0,
                "trace_loss": 0,
                "hallucinated_step_loss": 0,
            }
            while not done:
                num_action += 1
                action_idx = self._select_action(state, num_action)
                reward, next_state, done = self._step(state, action_idx, num_action)

                self._add_to_buffers(
                    prev_state,
                    prev_action_idx,
                    prev_reward,
                    state,
                    action_idx,
                    reward,
                    next_state,
                    done,
                    num_action,
                )

                prev_state = state
                prev_action_idx = action_idx
                prev_reward = reward

                state = next_state

            self._decrement_epsilon()

            if self.reward_type == "approximate":
                self._update_estimator(losses)

            self._learn(losses)

            traditional_losses.append(
                losses["traditional_loss"] / num_action
                if losses["traditional_loss"] > 0
                else 0
            )
            commutative_losses.append(
                losses["commutative_loss"] / num_action
                if losses["commutative_loss"] > 0
                else 0
            )
            hallucinated_losses.append(
                losses["hallucinated_loss"] / num_action
                if losses["hallucinated_loss"] > 0
                else 0
            )
            step_losses.append(
                losses["step_loss"] / num_action if losses["step_loss"] > 0 else 0
            )
            trace_losses.append(
                losses["trace_loss"] / num_action if losses["trace_loss"] > 0 else 0
            )
            hallucinated_step_losses.append(
                losses["hallucinated_step_loss"] / num_action
                if losses["hallucinated_step_loss"] > 0
                else 0
            )

            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window :])
            avg_commutative_losses = np.mean(commutative_losses[-self.sma_window :])
            avg_hallucinated_losses = np.mean(hallucinated_losses[-self.sma_window :])
            avg_step_loss = np.mean(step_losses[-self.sma_window :])
            avg_trace_losses = np.mean(trace_losses[-self.sma_window :])
            avg_hallucinated_step_losses = np.mean(
                hallucinated_step_losses[-self.sma_window :]
            )

            wandb.log(
                {
                    "Average Traditional Loss": avg_traditional_losses,
                    "Average Commutative Loss": avg_commutative_losses,
                    "Average Hallucinated Loss": avg_hallucinated_losses,
                    "Average Step Loss": avg_step_loss,
                    "Average Trace Loss": avg_trace_losses,
                    "Average Hallucinated Step Loss": avg_hallucinated_step_losses,
                },
                step=episode,
            )

        best_set = list(map(int, best_set))
        return best_return, best_set

    def generate_target_sum(self, problem_instance: str) -> np.ndarray:
        self.epsilon = 1
        self.num_updates = 0
        self.normal_traces = 0
        self.commutative_traces = 0
        self.hallucinated_traces = 0

        if self.target_sum == 0:
            self._set_problem(problem_instance)

        self.dqn = DQN(self.seed, self.max_elements, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        self.replay_buffer = ReplayBuffer(
            self.seed, self.max_elements, 1, self.buffer_size, self.action_dims
        )

        self.estimator = RewardEstimator(
            self.seed, self.step_dims, self.estimator_alpha
        )
        self.reward_buffer = RewardBuffer(
            self.seed,
            self.step_dims,
            self.max_elements,
            self.estimator_buffer_size,
            self.action_dims,
        )

        self._init_wandb(problem_instance)

        best_return, best_set = self._train()

        found_sum = sum(best_set) == self.target_sum

        wandb.log(
            {
                "Best Set": best_set,
                "Return": best_return,
                "Total Updates": self.num_updates,
                "Normal Traces": self.normal_traces,
                "Commutative Traces": self.commutative_traces,
                "Hallucinated Traces": self.hallucinated_traces,
                "Found Sum": found_sum,
            }
        )

        wandb.finish()

        return best_set
