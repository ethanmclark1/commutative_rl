import argparse


def get_arguments() -> tuple:
    parser = argparse.ArgumentParser(
        description="Teach a multi-agent system to create an emergent language."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generation {default_val: 0}",
    )

    parser.add_argument(
        "--n_agents",
        type=int,
        default=1,
        help="Number of agents in the environment {default_val: 1}",
    )

    parser.add_argument(
        "--n_large_obstacles",
        type=int,
        default=5,
        help="Number of large obstacles in the environment (no more than 16) {default_val: 2}",
    )

    parser.add_argument(
        "--n_small_obstacles",
        type=int,
        default=4,
        help="Number of small obstacles in the environment {default_val: 10}",
    )

    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        default=["TraditionalDQN"],
        choices=[
            "TraditionalDQN",
            "CommutativeDQN",
        ],
        help="Choose which approach to use {default_val: basic_dqn, choices: [%(choices)s]}",
    )

    parser.add_argument(
        "--problem_instances",
        type=str,
        nargs="+",
        default=["bisect"],
        choices=[
            "bisect",
            "circle",
            "cross",
            "corners",
            "staggered",
            "quarters",
            "scatter",
            "stellaris",
        ],
        help="Which problem(s) to attempt {default_val: %(default), choices: [%(choices)s]}",
    )

    parser.add_argument(
        "--n_episode_steps",
        type=int,
        default=12,
        help="Number of steps {default_val: %(default)}",
    )

    parser.add_argument(
        "--granularity",
        type=float,
        default=None,
        help="Granularity of the candidate lines {default_val: %(default)}",
    )

    parser.add_argument(
        "--safe_area_multiplier",
        type=float,
        default=None,
        help="Multiplier for safe area {default_val: %(default)}",
    )

    parser.add_argument(
        "--failed_path_cost",
        type=float,
        default=None,
        help="Cost for failed path {default_val: %(default)}",
    )

    parser.add_argument(
        "--configs_to_consider",
        type=int,
        default=None,
        help="Number of configurations to consider {default_val: %(default)}",
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

    args = parser.parse_args()

    return (
        args.seed,
        args.n_agents,
        args.n_large_obstacles,
        args.n_small_obstacles,
        args.approaches,
        args.problem_instances,
        args.n_episode_steps,
        args.granularity,
        args.safe_area_multiplier,
        args.failed_path_cost,
        args.configs_to_consider,
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
