import argparse

def get_arguments() -> tuple:
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.'
        )
    
    parser.add_argument(
        '--num_agents', 
        type=int, 
        default=1,
        help='Number of agents in the environment {default_val: 1}'
        )
    
    parser.add_argument(
        '--num_large_obstacles',
        type=int, 
        default=5,
        help='Number of large obstacles in the environment (no more than 16) {default_val: 2}'
        )
    
    parser.add_argument(
        '--num_small_obstacles', 
        type=int, 
        default=4,
        help='Number of small obstacles in the environment {default_val: 10}'
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
        choices=['basic_dqn', 'commutative_dqn', 'hallucinated_dqn', 'basic_sac', 'commutative_sac'],
        help='Choose which approach to use {default_val: basic_dqn, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--problem_instances', 
        type=str, 
        nargs='+',
        default=['bisect'], 
        choices=['bisect', 'circle', 'cross', 'corners', 'staggered', 'quarters', 'scatter', 'stellaris'],
        help='Which problem(s) to attempt {default_val: cross, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--reward_type',
        type=str,
        default='true',
        choices=['true', 'approximate'],
        help='Which reward prediction type to use {default_val: true, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--render_mode', 
        type=str, 
        default='None', 
        choices=['human', 'rgb_array', 'None'], 
        help='Mode of visualization {default_val: None, choices: [%(choices)s]}'
        )
    
    args = parser.parse_args()
        
    return args.num_agents, args.num_large_obstacles, args.num_small_obstacles, args.seed, args.names, args.problem_instances, args.reward_type, args.render_mode