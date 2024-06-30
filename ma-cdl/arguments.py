import argparse

def get_arguments() -> tuple:
    parser = argparse.ArgumentParser(
        description='Try to find the target set.'
        )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for random number generation {default_val: 0}'
        )
    
    parser.add_argument(
        '--names', 
        type=str, 
        nargs='+',
        default=['commutative_dqn'], 
        choices=['basic_dqn', 'commutative_dqn', 'hallucinated_dqn'],
        help='Choose which approach to use {default_val: basic_dqn, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--num_sets',
        type=int,
        default=4,
        help='Size of set {default_val: 8}'
    )
    
    parser.add_argument(
        '--max_action',
        type=int,
        default=8,
        help='Size of set {default_val: 8}'
    )
    
    parser.add_argument(
        '--action_dims',
        type=int,
        default=25,
        help='Size of action space {default_val: 8}'
    )
    
    parser.add_argument(
        '--problem_instances', 
        type=str, 
        nargs='+',
        default=['instance_0'], 
        choices=['instance_0', 'instance_1', 'instance_2', 'instance_3'],
        help='Which problem(s) to attempt {default_val: cross, choices: [%(choices)s]}'
        )

    args = parser.parse_args()
        
    return args.seed, args.names, args.num_sets, args.max_action, args.action_dims, args.problem_instances