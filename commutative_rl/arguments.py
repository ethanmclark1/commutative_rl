import argparse


def parse_num_instances() -> tuple:
    parser = argparse.ArgumentParser(description="Initial argument parser.")

    parser.add_argument(
        "--num_instances",
        type=int,
        default=30,
        help="Number of instances to generate dynamically.",
    )

    args, remaining_argv = parser.parse_known_args()

    return args.num_instances, remaining_argv


def get_arguments(num_instances: int, remaining_argv: list) -> tuple:
    instance_choices = [f"instance_{i}" for i in range(num_instances)]

    parser = argparse.ArgumentParser(description="Try to find the target set.")

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generation {default_val: 0}",
    )

    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        default=["Commutative"],
        choices=["Traditional", "Commutative", "Hallucinated"],
        help="Choose which approach to use {default_val: basic_dqn, choices: [%(choices)s]}",
    )

    parser.add_argument(
        "--max_sum", type=int, default=100, help="Size of set {default_val: %(default)}"
    )

    parser.add_argument(
        "--action_dims",
        type=int,
        default=25,
        help="Size of action space {default_val: %(default)}",
    )

    parser.add_argument(
        "--negative_actions",
        action="store_true",
        default=True,
        help="Allow negative actions {default_val: False}",
    )

    parser.add_argument(
        "--duplicate_actions",
        action="store_true",
        default=True,
        help="Allow duplicate actions {default_val: False}",
    )

    parser.add_argument(
        "--problem_instances",
        type=str,
        nargs="+",
        default=["instance_26"],
        choices=instance_choices,
        help="Which problem(s) to attempt {default_val: %(default)s, choices: [%(choices)s]}",
    )

    parser.add_argument(
        "--reward_type",
        type=str,
        default="true",
        choices=["true", "approximate"],
        help="Type of reward to use {default_val: basic, choices: [%(choices)s]}",
    )

    parser.add_argument(
        "--reward_noise",
        type=float,
        default=0.00,
        help="Variance of reward noise {default_val: %(default)s}",
    )

    args = parser.parse_args(remaining_argv)

    return (
        args.seed,
        args.approaches,
        args.max_sum,
        args.action_dims,
        args.negative_actions,
        args.duplicate_actions,
        args.problem_instances,
        args.reward_type,
        args.reward_noise,
    )
