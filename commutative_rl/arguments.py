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
        default=["Traditional"],
        choices=[
            "TraditionalQTable",
            "CommutativeQTable",
            "TripleTraditionalQTable",
            "TraditionalDQN",
            "CommutativeDQN",
            "CommutativeIndependentSamplesDQN",
            "TripleTraditionalDQN",
        ],
        help="Choose which approach to use {default_val: basic_dqn, choices: [%(choices)s]}",
    )

    parser.add_argument(
        "--min_sum_range",
        type=int,
        default=9000,
        help="Minimum bound on sum range {default_val: %(default)}",
    )

    parser.add_argument(
        "--max_sum_range",
        type=int,
        default=11000,
        help="Maximum bound on sum range {default_val: %(default)}",
    )

    parser.add_argument(
        "--min_elem_range",
        type=int,
        default=80,
        help="Minimum bound on element range {default_val: %(default)}",
    )

    parser.add_argument(
        "--max_elem_range",
        type=int,
        default=300,
        help="Maximum bound on element range {default_val: %(default)}",
    )

    parser.add_argument(
        "--n_elems",
        type=int,
        default=21,
        help="Number of elements in the problem {default_val: %(default)}",
    )

    parser.add_argument(
        "--problem_instances",
        type=str,
        nargs="+",
        default=[
            "instance_6",
        ],
        choices=instance_choices,
        help="Which problem(s) to attempt {default_val: %(default)s, choices: [%(choices)s]}",
    )

    parser.add_argument(
        "--max_noise",
        type=int,
        default=10,
        help="Variance of reward noise {default_val: %(default)s}",
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
        "--target_update_freq",
        type=int,
        default=None,
        help="Frequency of target network update {default_val: %(default)}",
    )

    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=None,
        help="Gradient clipping norm {default_val: %(default)}",
    )

    parser.add_argument(
        "--loss_fn",
        type=str,
        default=None,
        choices=["Huber", "MSE"],
        help="Loss function to use {default_val: %(default)}",
    )

    parser.add_argument(
        "--layer_norm",
        type=int,
        default=None,
        help="Use layer normalization (default: %(default)s)",
    )

    parser.add_argument(
        "--aggregation_type",
        type=str,
        default=None,
        choices=[
            "equal",
            "trace_front",
            "trace_back",
            "mirror_front",
            "mirror_back",
            "true_reward",
        ],
    )

    args = parser.parse_args(remaining_argv)

    return (
        args.seed,
        args.approaches,
        args.min_sum_range,
        args.max_sum_range,
        args.min_elem_range,
        args.max_elem_range,
        args.n_elems,
        args.problem_instances,
        args.max_noise,
        args.alpha,
        args.epsilon,
        args.gamma,
        args.batch_size,
        args.buffer_size,
        args.hidden_dims,
        args.target_update_freq,
        args.grad_clip_norm,
        args.loss_fn,
        args.layer_norm,
        args.aggregation_type,
    )
