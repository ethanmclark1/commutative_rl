import argparse


def parse_num_instances() -> tuple:
    parser = argparse.ArgumentParser(description="Initial argument parser.")

    parser.add_argument(
        "--num_instances",
        type=int,
        default=10,
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

    instance_choices = [f"instance_{i}" for i in range(num_instances)]
    parser.add_argument(
        "--problem_instances",
        type=str,
        nargs="+",
        default=["instance_2"],
        choices=instance_choices,
        help="Which problem to attempt {default_val: %(default)s, choices: [%(choices)s]}",
    )

    parser.add_argument(
        "--noise_type",
        type=str,
        default="none",
        choices=["none", "residents", "full"],
        help="Type of noise to add into the environment {default_val: %(default)s, choices: [%(choices)s]}",
    )

    args = parser.parse_args(remaining_argv)

    return (
        args.seed,
        args.approaches,
        args.problem_instances,
        args.noise_type,
    )
