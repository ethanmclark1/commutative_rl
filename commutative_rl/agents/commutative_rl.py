import copy
import wandb
import torch
import numpy as np
import more_itertools

from env import Env
from .utils.networks import DQN, RewardEstimator
from .utils.buffers import encode, ReplayBuffer, RewardBuffer, CommutativeRewardBuffer


class Parent(Env):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        max_elements: int,
        action_dims: int,
        negative_actions: bool,
        duplicate_actions: bool,
        reward_type: str,
        reward_noise: float,
    ) -> None:

        super(Parent, self).__init__(
            seed,
            num_instances,
            max_elements,
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
        self.estimator_alpha = 0.004
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
        self.sma_window = 25 if self.reward_noise == 0 else 250

        # Evaluation
        self.eval_freq = 1
        self.eval_window = 25 if self.reward_noise == 0 else 250

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
            state = encode(state, self.action_dims)
            num_action = encode(num_action - 1, self.max_elements)

            state = torch.FloatTensor(state)
            num_action = torch.FloatTensor([num_action])

            with torch.no_grad():
                action_idx = self.dqn(state, num_action).argmax().item()
        else:
            action_idx = self.action_rng.integers(self.action_dims)

        return action_idx

    def _add_to_buffers(
        self,
        state: list,
        action_idx: int,
        reward: float,
        next_state: list,
        done: bool,
        num_action: int,
        prev_state: list = None,
        prev_action: int = None,
        prev_reward: float = None,
    ) -> None:

        self.normal_traces += 1
        self.replay_buffer.add(state, action_idx, reward, next_state, done, num_action)

        if self.reward_type == "approximate":
            self.reward_buffer.add(state, action_idx, reward, next_state, num_action)

    def _update_estimator(
        self, losses: dict, reward_buffer: object = None, loss_type: str = None
    ) -> None:
        if (
            reward_buffer is None
            or loss_type is None
            or self.estimator_batch_size > reward_buffer.real_size
        ):
            return

        steps, rewards = reward_buffer.sample(self.estimator_batch_size)
        r_pred = self.estimator(steps)

        self.estimator.optim.zero_grad()
        step_loss = self.estimator.loss(r_pred, rewards)
        step_loss.backward()
        self.estimator.optim.step()

        losses[loss_type] += abs(step_loss.item())

    def _learn(
        self, losses: dict, replay_buffer: object = None, loss_type: str = None
    ) -> np.ndarray:
        if (
            replay_buffer is None
            or loss_type is None
            or self.batch_size > replay_buffer.real_size
        ):
            return None

        indices = replay_buffer.sample(self.batch_size)

        state = replay_buffer.state[indices]
        action_idx = replay_buffer.action_idx[indices]
        reward = replay_buffer.reward[indices]
        next_state = replay_buffer.next_state[indices]
        done = replay_buffer.done[indices]
        num_action = replay_buffer.num_action[indices]
        next_num_action = num_action + self.num_action_increment

        if self.reward_type == "approximate":
            action_enc = encode(action_idx, self.action_dims)
            features = torch.cat([state, action_enc, next_state, num_action], dim=-1)

            with torch.no_grad():
                reward = self.estimator(features)

        q_values = self.dqn(state, num_action)
        selected_q_values = torch.gather(q_values, 1, action_idx)
        next_q_values = self.target_dqn(next_state, next_num_action)
        target_q_values = reward + ~done * torch.max(next_q_values, dim=1).values.view(
            -1, 1
        )

        self.num_updates += 1
        self.dqn.optim.zero_grad()
        loss = self.dqn.loss(selected_q_values, target_q_values)
        loss.backward()
        self.dqn.optim.step()

        for target_param, local_param in zip(
            self.target_dqn.parameters(), self.dqn.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

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
                    state,
                    action_idx,
                    reward,
                    next_state,
                    done,
                    num_action,
                    prev_state,
                    prev_action_idx,
                    prev_reward,
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


class Traditional(Parent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        max_elements: int,
        action_dims: int,
        negative_actions: bool,
        duplicate_actions: bool,
        reward_type: str,
        reward_noise: float,
    ) -> None:

        super(Traditional, self).__init__(
            seed,
            num_instances,
            max_elements,
            action_dims,
            negative_actions,
            duplicate_actions,
            reward_type,
            reward_noise,
        )

    def _update_estimator(
        self, losses: dict, reward_buffer: object = None, loss_type: str = None
    ) -> None:
        super()._update_estimator(losses, self.reward_buffer, "step_loss")
        super()._update_estimator(losses, self.reward_buffer, "step_loss")

    def _learn(
        self, losses: dict, replay_buffer: object = None, loss_type: str = None
    ) -> None:
        super()._learn(losses, self.replay_buffer, "traditional_loss")
        super()._learn(losses, self.replay_buffer, "traditional_loss")


class Commutative(Parent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        max_elements: int,
        action_dims: int,
        negative_actions: bool,
        duplicate_actions: bool,
        reward_type: str,
        reward_noise: float,
    ) -> None:

        super(Commutative, self).__init__(
            seed,
            num_instances,
            max_elements,
            action_dims,
            negative_actions,
            duplicate_actions,
            reward_type,
            reward_noise,
        )

        self.commutive_replay_buffer = None
        self.commutative_reward_buffer = None

    def _add_to_buffers(
        self,
        state: list,
        action_idx: int,
        reward: float,
        next_state: list,
        done: bool,
        num_action: int,
        prev_state: list,
        prev_action: int,
        prev_reward: float,
    ) -> None:

        super()._add_to_buffers(state, action_idx, reward, next_state, done, num_action)

        if num_action % 2 == 0 and self.actions[action_idx] != 0:
            commutative_state = self._get_next_state(prev_state, action_idx)
            if self.reward_type == "true":
                prev_commutative_reward, commutative_state, done = self._step(
                    prev_state, action_idx, num_action - 1
                )
                self.commutative_replay_buffer.add(
                    prev_state,
                    action_idx,
                    prev_commutative_reward,
                    commutative_state,
                    done,
                    num_action - 1,
                )

                commutative_reward, next_state, done = self._step(
                    commutative_state, prev_action, num_action
                )
                self.commutative_replay_buffer.add(
                    commutative_state,
                    prev_action,
                    commutative_reward,
                    next_state,
                    done,
                    num_action,
                )

                self.commutative_traces += 2
            else:
                self.commutative_replay_buffer.add(
                    prev_state, action_idx, -1, commutative_state, done, num_action - 1
                )
                self.commutative_replay_buffer.add(
                    commutative_state, prev_action, -1, next_state, done, num_action
                )

                self.commutative_reward_buffer.add(
                    prev_state,
                    action_idx,
                    prev_reward,
                    commutative_state,
                    prev_action,
                    reward,
                    next_state,
                    num_action,
                )

                self.commutative_traces += 2

    def _update_estimator(
        self, losses: dict, reward_buffer: object = None, loss_type: str = None
    ) -> None:
        super()._update_estimator(losses, self.reward_buffer, "step_loss")

        if self.estimator_batch_size > self.commutative_reward_buffer.real_size:
            return

        steps, rewards = self.commutative_reward_buffer.sample(
            self.estimator_batch_size
        )
        r2_pred = self.estimator(steps[:, 0])
        r3_pred = self.estimator(steps[:, 1])

        self.estimator.optim.zero_grad()
        loss_r2 = self.estimator.loss(r2_pred + r3_pred.detach(), rewards)
        loss_r3 = self.estimator.loss(r2_pred.detach() + r3_pred, rewards)
        trace_loss = loss_r2 + loss_r3
        trace_loss.backward()
        self.estimator.optim.step()

        losses["trace_loss"] += abs(trace_loss.item() / 2)

    def _learn(
        self, losses: dict, replay_buffer: object = None, loss_type: str = None
    ) -> None:
        super()._learn(losses, self.replay_buffer, "traditional_loss")
        super()._learn(losses, self.commutative_replay_buffer, "commutative_loss")

    def generate_target_sum(self, problem_instance: str) -> np.ndarray:
        self._set_problem(problem_instance)

        self.commutative_replay_buffer = ReplayBuffer(
            self.seed, self.max_elements, 1, self.buffer_size, self.action_dims
        )
        self.commutative_reward_buffer = CommutativeRewardBuffer(
            self.seed,
            self.step_dims,
            self.max_elements,
            self.estimator_buffer_size,
            self.action_dims,
        )

        super().generate_target_sum(problem_instance)


class Hallucinated(Parent):
    def __init__(
        self,
        seed: int,
        num_instances: int,
        max_elements: int,
        action_dims: int,
        negative_actions: int,
        duplicate_actions: int,
        reward_type: str,
        reward_noise: float,
    ) -> None:

        super(Hallucinated, self).__init__(
            seed,
            num_instances,
            max_elements,
            action_dims,
            negative_actions,
            duplicate_actions,
            reward_type,
            reward_noise,
        )

        self.hallucinated_replay_buffer = None
        self.hallucinated_reward_buffer = None

    def _hallucinate(self, non_zero_elements: list, with_approximate: bool) -> None:
        powerset = list(more_itertools.powerset(non_zero_elements))

        for subset in powerset:
            state = sorted(list(subset)) + [0.0] * (self.max_elements - len(subset))
            actions = [
                element for element in non_zero_elements if element not in subset
            ]

            if len(subset) != self.max_elements:
                actions.insert(0, 0.0)

            num_action = len(subset) + 1
            for action in actions:
                action_idx = self.actions.index(action)
                reward, next_state, done = self._step(state, action_idx, num_action)

                self.hallucinated_replay_buffer.add(
                    state, action_idx, reward, next_state, done, num_action
                )
                if with_approximate:
                    self.hallucinated_reward_buffer.add(
                        state, action_idx, reward, next_state, num_action
                    )

    def _add_to_buffers(
        self,
        state: list,
        action_idx: int,
        reward: float,
        next_state: list,
        done: bool,
        num_action: int,
        prev_state: list,
        prev_action: int,
        prev_reward: float,
    ) -> None:

        super()._add_to_buffers(state, action_idx, reward, next_state, done, num_action)

        if done:
            with_approximate = self.reward_type == "approximate"
            non_zero_elements = [element for element in next_state if element != 0]
            if len(non_zero_elements) > 0:
                self._hallucinate(non_zero_elements, with_approximate)

    def _update_estimator(
        self, losses: dict, reward_buffer: object = None, loss_type: str = None
    ) -> None:
        super()._update_estimator(losses, self.reward_buffer, "step_loss")
        super()._update_estimator(
            losses, self.hallucinated_reward_buffer, "hallucinated_step_loss"
        )

    def _learn(
        self, losses: dict, replay_buffer: object = None, loss_type: str = None
    ) -> None:
        super()._learn(losses, self.replay_buffer, "traditional_loss")
        super()._learn(losses, self.hallucinated_replay_buffer, "hallucinated_loss")

    def generate_target_sum(self, problem_instance: str) -> np.ndarray:
        self._set_problem(problem_instance)

        self.hallucinated_replay_buffer = ReplayBuffer(
            self.seed,
            self.max_elements,
            1,
            self.hallucinated_buffer_size,
            self.action_dims,
        )
        self.hallucinated_reward_buffer = RewardBuffer(
            self.seed,
            self.step_dims,
            self.max_elements,
            self.estimator_hallucinated_buffer_size,
            self.action_dims,
        )

        super().generate_target_sum(problem_instance)
