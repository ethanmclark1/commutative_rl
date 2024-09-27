import os
import sys
import unittest
import numpy as np
import itertools

current_dir = os.path.dirname(os.path.abspath(__file__))
Agent_dir = os.path.dirname(current_dir)
sys.path.insert(0, Agent_dir)

from env import Env


class TestCommutative(unittest.TestCase):
    def __init__(self, methodName: str = "runTest", **kwargs):
        super().__init__(methodName)

        self.seed = kwargs["seed"]
        self.num_instances = kwargs["num_instances"]
        self.max_sum = kwargs["max_sum"]
        self.action_dims = kwargs["action_dims"]
        self.reward_type = kwargs["reward_type"]
        self.reward_noise = kwargs["reward_noise"]

        self.actions_rng = np.random.default_rng(self.seed)
        self.env = Env(
            self.seed,
            self.num_instances,
            self.max_sum,
            self.action_dims,
            self.reward_type,
            self.reward_noise,
        )

    def _apply_actions(self, final_state: list):
        episodic_return = 0
        state, num_action, _ = self.env._generate_start_state()

        for action in final_state:
            num_action += 1
            reward, next_state, _ = self.env._step(state, action, num_action)
            state = next_state

            episodic_return += reward

        return episodic_return, state

    def _generate_valid_action_sets(self, num_tests: int) -> list:
        action_sets = []

        valid_actions = [
            action
            for action in range(1, self.action_dims + 1)
            if action not in self.env.invalid_actions
        ]

        for _ in range(num_tests):
            num_elements = self.actions_rng.integers(
                5, min(len(valid_actions), self.env.max_elements)
            )
            final_state = sorted(
                self.actions_rng.choice(valid_actions, size=num_elements, replace=False)
            )
            action_sets.append(final_state)

        return action_sets

    def test_commutative(self, num_tests: int = 25):
        action_sets = self._generate_valid_action_sets(num_tests)

        for num_instance in range(self.num_instances):
            self.env.target_sum = self.env._get_target_sum(f"instance_{num_instance}")

            for action_set in action_sets:
                final_state = action_set + [0] * (
                    self.env.max_elements - len(action_set)
                )
                original_return, original_state = self._apply_actions(final_state)
                permutations = itertools.permutations(action_set)

                for perm in permutations:
                    final_state = list(perm) + [0] * (self.env.max_elements - len(perm))
                    perm_return, perm_state = self._apply_actions(perm)

                    self.assertEqual(
                        original_state,
                        perm_state,
                        f"States differ for permutation {perm}",
                    )
                    self.assertAlmostEqual(
                        original_return,
                        perm_return,
                        places=5,
                        msg=f"Returns differ for permutation {perm}",
                    )

                print(
                    f"Commutativity test passed for final state: {original_state} with final return: {original_return}"
                )


def run_tests(**kwargs):
    suite = unittest.TestSuite()
    suite.addTest(TestCommutative("test_commutative", **kwargs))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    run_tests(
        seed=42,
        num_instances=30,
        max_sum=100,
        action_dims=30,
        reward_type="true",
        reward_noise=0.0,
    )
