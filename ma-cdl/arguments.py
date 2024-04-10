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
        default=8,
        help='Number of large obstacles in the environment (no more than 16) {default_val: 2}'
        )
    
    parser.add_argument(
        '--num_small_obstacles', 
        type=int, 
        default=4,
        help='Number of small obstacles in the environment {default_val: 10}'
        )
    
    parser.add_argument(
        '--approach', 
        type=str, 
        default='commutative_dqn', 
        choices=['basic_dqn', 'commutative_dqn', 'basic_td3', 'commutative_td3'],
        help='Choose which approach to use {default_val: basic_dqn, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--problem_instance', 
        type=str, 
        default='circle', 
        choices=['bisect', 'circle', 'cross', 'corners', 'staggered', 'quarters', 'scatter', 'stellaris'],
        help='Which problem to attempt {default_val: cross, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=1, 
        choices=[0, 1], 
        help='Generate a random initial state for the agent {default_val: None, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--train_type',
        type=str,
        default='online',
        choices=['online', 'offline'],
        help='Whether the model is trained online or offline {default_val: train, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--reward_type',
        type=str,
        default='approximate',
        choices=['true', 'approximate'],
        help='Which reward prediction type to use {default_val: basic, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--render_mode', 
        type=str, 
        default='None', 
        choices=['human', 'rgb_array', 'None'], 
        help='Mode of visualization {default_val: None, choices: [%(choices)s]}'
        )
    
    args = parser.parse_args()
        
    return args.num_agents, args.num_large_obstacles, args.num_small_obstacles, args.approach, args.problem_instance, bool(args.random_state), args.train_type, args.reward_type, args.render_mode