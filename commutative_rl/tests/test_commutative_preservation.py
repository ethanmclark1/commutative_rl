import os
import sys
import yaml
import torch
import unittest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from env import Env
from agents.utils.networks import DuelingDQN
from arguments import parse_num_instances, get_arguments


class TestCommutativePreservation(unittest.TestCase):
    def setUp(self) -> None:
        cwd = os.getcwd()
        ckpt_dir = os.path.join(cwd, "commutative_rl", "ckpt", self.problem_instance)
        filepath = os.path.join(cwd, "commutative_rl", "agents", "utils", "config.yaml")

        with open(filepath, "r") as file:
            config = yaml.safe_load(file)

        if self.hidden_dims is None:
            self.hidden_dims = config["agent"]["dqn"]["hidden_dims"]
        if self.layer_norm is None:
            self.layer_norm = config["agent"]["dqn"]["layer_norm"]

        self.env = Env(
            self.seed,
            self.num_instances,
            self.sum_range,
            self.elem_range,
            self.n_elems,
            self.max_noise,
            config["env"],
        )
        self.env.set_problem(self.problem_instance)

        self.q_table = np.load(os.path.join(ckpt_dir, "CommutativeQTable.npy"))

        self.network = DuelingDQN(
            seed=self.seed,
            state_dims=self.env.n_statistics,
            action_dims=self.n_elems,
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
        )
        self.network.eval()
        state_dict = torch.load(
            os.path.join(ckpt_dir, "CommutativeIndependentSamplesDQN.pt"),
            weights_only=True,
        )
        self.network.load_state_dict(state_dict)

        self.rng = np.random.default_rng(self.seed)

    # generate random action groups for testing
    def _generate_test_action_groups(self, elem_arr: np.ndarray) -> list:
        sequences = []

        max_elem = elem_arr.max()

        count = 0
        while count < self.n_groups:
            seq_len = self.rng.integers(2, self.env.n_steps)
            elements = self.rng.choice(elem_arr, size=seq_len)

            # ensure that sum of the sequence is within the valid range
            max_valid_sum = (
                self.env.target_sum - 2 * max_elem - self.max_noise * seq_len
            )

            if sum(elements) < max_valid_sum:
                sequences.append(elements)
                count += 1

        return sequences

    def _get_initial_state(self, action_group: np.ndarray) -> torch.Tensor:
        state = np.zeros(self.env.n_statistics)

        count = sum(
            [action + self.rng.integers(0, self.max_noise) for action in action_group]
        )

        state[0] = count / self.env.target_sum
        state[1] = len(action_group) / self.env.n_steps

        return state

    def _get_next_state(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        new_elem = self.env.elements[action_idx]

        current_sum = int(state[0] * self.env.target_sum)
        current_n_steps = int(state[1] * self.env.n_steps)

        new_sum = (current_sum + new_elem) / self.env.target_sum
        new_n_steps = (current_n_steps + 1) / self.env.n_steps

        next_state = np.array([new_sum, new_n_steps], dtype=float)

        return next_state

    def test_commutative_preservation(self) -> None:
        qtable_diffs = []
        dqn_diffs = []

        elem_arr = np.array(self.env.elements)[1:]
        action_groups = self._generate_test_action_groups(elem_arr)

        for action_group in action_groups:
            state = self._get_initial_state(action_group)
            episode_step = len(action_group)

            action_a_idx = self.rng.choice(self.n_elems)
            action_b_idx = self.rng.choice(self.n_elems)

            state_1 = self.env._get_next_state(state, action_a_idx)
            next_state = self.env._get_next_state(state_1, action_b_idx)

            sum1 = int(next_state[0] * self.env.target_sum)

            qtable_val1 = self.q_table[sum1].mean()
            with torch.no_grad():
                dqn_val1 = (
                    self.network(torch.tensor(next_state, dtype=torch.float32))
                    .mean()
                    .item()
                )

            state_2 = self._get_next_state(state, action_a_idx)
            next_state = self._get_next_state(state_2, action_b_idx)

            sum2 = int(next_state[0] * self.env.target_sum)

            qtable_val2 = self.q_table[sum2].mean()
            with torch.no_grad():
                dqn_val2 = (
                    self.network(torch.tensor(next_state, dtype=torch.float32))
                    .mean()
                    .item()
                )

            qtable_diffs.append(abs(qtable_val1 - qtable_val2))
            dqn_diffs.append(abs(dqn_val1 - dqn_val2))

        avg_qtable_diff = np.mean(qtable_diffs)
        avg_dqn_diff = np.mean(dqn_diffs)

        print(f"\nProblem Instance: {self.problem_instance}")
        print(f"Average Q-table difference: {avg_qtable_diff:.6f}")
        print(f"Average DQN difference: {avg_dqn_diff:.6f}")
        print(
            f"DQN/Q-table difference ratio: {avg_dqn_diff/avg_qtable_diff if avg_qtable_diff > 0 else float('inf'):.2f}"
        )


def run_tests():
    ckpt_dir = os.path.join(os.getcwd(), "commutative_rl", "ckpt")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError("No checkpoint directory")

    # check if all necessary files are present for each problem instance
    problem_instances = os.listdir(ckpt_dir)
    for problem_instance in problem_instances:
        if not (
            os.path.isfile(
                os.path.join(ckpt_dir, problem_instance, "CommutativeQTable.npy")
            )
            and os.path.isfile(
                os.path.join(
                    ckpt_dir, problem_instance, "CommutativeIndependentSamplesDQN.pt"
                )
            )
        ):
            problem_instances.remove(problem_instance)

    num_instances, remaining_argv = parse_num_instances()
    args = get_arguments(num_instances, remaining_argv)

    suite = unittest.TestSuite()

    for problem_instance in problem_instances:
        test_method_name = f"test_commutative_preservation_{problem_instance}"

        setattr(
            TestCommutativePreservation,
            test_method_name,
            TestCommutativePreservation.test_commutative_preservation,
        )

        test_case = TestCommutativePreservation(methodName=test_method_name)
        test_case.seed = args[0]
        test_case.num_instances = num_instances
        test_case.sum_range = range(args[2], args[3])
        test_case.elem_range = range(args[4], args[5])
        test_case.n_elems = args[6]
        test_case.problem_instance = problem_instance
        test_case.max_noise = args[8]
        test_case.hidden_dims = args[14]
        test_case.layer_norm = args[18]

        test_case.n_groups = 100

        suite.addTest(test_case)

    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    run_tests()
