import argparse

def parse_num_instances() -> tuple:
    parser = argparse.ArgumentParser(description='Initial argument parser.')
    
    parser.add_argument(
        '--num_instances',
        type=int, 
        default=40, 
        help='Number of instances to generate dynamically.'
        )
    
    args, remaining_argv = parser.parse_known_args()
    
    return args.num_instances, remaining_argv

def get_arguments(num_instances: int, remaining_argv: list) -> tuple:
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.'
        )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for the random number generator {default_val: %(default)s}'
    )
    
    parser.add_argument(
        '--approaches', 
        type=str, 
        nargs='+',
        default=['BasicDQN'], 
        choices=['BasicDQN', 'CommutativeDQN', 'HallucinatedDQN'],
        help='Choose which approach to use {default_val: basic_dqn, choices: [%(choices)s]}'
        )
    
    instance_choices = [f'instance_{i}' for i in range(num_instances)]
    parser.add_argument(
        '--problem_instances', 
        type=str, 
        nargs='+',
        default=['instance_2'],
        choices=instance_choices,
        help='Which problem to attempt (instance > 30 is 8x8 problem size) {default_val: %(default)s, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--reward_type', 
        type=str, 
        default='approximate', 
        choices=['true', 'approximate'], 
        help='Type of way to predict the reward r_3 {default_val: %(default)s}'
        )
    
    parser.add_argument(
        '--noise_type',
        type=str,
        default='full',
        choices=['residents', 'full'],
        help='Type of noise to add into the environment {default_val: %(default)s, choices: [%(choices)s]}'
    )
    
    args = parser.parse_args(remaining_argv)
        
    return args.seed, args.approaches, args.problem_instances, args.reward_type, args.noise_type