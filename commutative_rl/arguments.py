import argparse


def parse_n_instances() -> tuple:
    parser = argparse.ArgumentParser(description="Initial argument parser.")

    parser.add_argument(
        "--n_instances",
        type=int,
        default=8,
        help="Number of instances to generate dynamically.",
    )

    args, remaining_argv = parser.parse_known_args()

    return args.n_instances, remaining_argv


def get_arguments(n_instances: int, remaining_argv: list) -> tuple:
    instance_choices = [f"instance_{i}" for i in range(n_instances)]

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
        default=["TraditionalDQN"],
        choices=[
            "TraditionalQTable",
            "TraditionalDQN",
            "CommutativeQTable",
            "CommutativeDQN",
            "TripleTraditionalQTable",
            "TripleTraditionalDQN",
        ],
        help="Choose which approach to use {default_val: basic_dqn, choices: [%(choices)s]}",
    )

    instance_choices = [f"instance_{i}" for i in range(n_instances)]
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
        default=None,
        choices=[None, "Residents", "Full"],
        help="Type of noise to add into the environment {default_val: %(default)s, choices: [%(choices)s]}",
    )

    parser.add_argument(
        "--grid_dims",
        type=str,
        default=None,
        choices=[None, "8x8", "16x16", "32x32"],
        help="Dimensions of the grid {default_val: %(default)}",
    )

    parser.add_argument(
        "--n_starts",
        type=int,
        default=None,
        help="Number of starting points {default_val: %(default)}",
    )

    parser.add_argument(
        "--n_goals",
        type=int,
        default=None,
        help="Number of goal points {default_val: %(default)}",
    )

    parser.add_argument(
        "--n_bridges",
        type=int,
        default=None,
        help="Number of bridges {default_val: %(default)}",
    )

    parser.add_argument(
        "--n_holes",
        type=int,
        default=None,
        help="Number of holes {default_val: %(default)}",
    )

    parser.add_argument(
        "--n_episode_steps",
        type=int,
        default=12,
        help="Number of steps {default_val: %(default)}",
    )

    parser.add_argument(
        "--configs_to_consider",
        type=int,
        default=None,
        help="Number of configurations to consider {default_val: %(default)}",
    )

    parser.add_argument(
        "--action_success_rate",
        type=float,
        default=None,
        help="Action success rate {default_val: %(default)}",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Learning rate {default_val: %(default)}",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Exploration rate {default_val: %(default)}",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor {default_val: %(default)}",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Size of batch {default_val: %(default)}",
    )

    parser.add_argument(
        "--buffer_size",
        type=int,
        default=None,
        help="Size of buffer {default_val: %(default)}",
    )

    parser.add_argument(
        "--hidden_dims",
        type=int,
        default=None,
        help="Size of hidden layer {default_val: %(default)}",
    )

    parser.add_argument(
        "--n_hidden_layers",
        type=int,
        default=None,
        help="Number of layers in the network {default_val: %(default)}",
    )

    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=None,
        help="Frequency of target network update {default_val: %(default)}",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate {default_val: %(default)}",
    )

    args = parser.parse_args(remaining_argv)

    return (
        args.seed,
        args.approaches,
        args.problem_instances,
        args.grid_dims,
        args.n_starts,
        args.n_goals,
        args.n_bridges,
        args.n_holes,
        args.n_episode_steps,
        args.noise_type,
        args.configs_to_consider,
        args.action_success_rate,
        args.alpha,
        args.epsilon,
        args.gamma,
        args.batch_size,
        args.buffer_size,
        args.hidden_dims,
        args.n_hidden_layers,
        args.target_update_freq,
        args.dropout,
    )
