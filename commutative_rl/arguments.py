import argparse

def parse_num_instances() -> tuple:
    parser = argparse.ArgumentParser(description='Initial argument parser.')
    
    parser.add_argument(
        '--num_instances',
        type=int, 
        default=30, 
        help='Number of instances to generate dynamically.'
        )
    
    args, remaining_argv = parser.parse_known_args()
    
    return args.num_instances, remaining_argv

def get_arguments(num_instances: int, remaining_argv: list) -> tuple:
    instance_choices = [f'instance_{i}' for i in range(num_instances)]
    
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
        '--approaches', 
        type=str, 
        nargs='+',
        default=['CommutativeDQN'], 
        choices=['BasicDQN', 'CommutativeDQN', 'HallucinatedDQN'],
        help='Choose which approach to use {default_val: basic_dqn, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--max_elements',
        type=int,
        default=10,
        help='Size of set {default_val: %(default)}'
        )
    
    parser.add_argument(
        '--action_dims',
        type=int,
        default=200,
        help='Size of action space {default_val: %(default)}'
        )
    
    parser.add_argument(
        '--problem_instances', 
        type=str, 
        nargs='+',
        default=['instance_2'], 
        choices=instance_choices,
        help='Which problem(s) to attempt {default_val: %(default)s, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--reward_type',
        type=str,
        default='approximate',
        choices=['true', 'approximate'],
        help='Type of reward to use {default_val: basic, choices: [%(choices)s]}'
    )
    
    parser.add_argument(
        '--reward_noise',
        type=float,
        default=1.00,
        help='Variance of reward noise {default_val: %(default)s}'
    )

    args = parser.parse_args(remaining_argv)
        
    return args.seed, args.approaches, args.max_elements, args.action_dims, args.problem_instances, args.reward_type, args.reward_noise