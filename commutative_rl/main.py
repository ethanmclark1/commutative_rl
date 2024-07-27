import itertools
import gymnasium as gym

from arguments import parse_num_instances, get_arguments
from agents.commutative_rl import BasicDQN, CommutativeDQN, HallucinatedDQN
    

if __name__ == '__main__':
    num_instances, remaining_argv = parse_num_instances()
    seed, approaches, problem_instances, random_state, train_type, reward_type, noise_type = get_arguments(num_instances, remaining_argv)
    
    instance_num = int(problem_instances.split('_')[-1])
    problem_size = 'big' if instance_num >= 30 else 'small'
    map_name = '8x8' if problem_size == 'big' else '4x4'
    
    env = gym.make(
        id='FrozenLake-v1', 
        map_name=map_name, 
        render_mode=None
        )
    
    approach_map = {
        'BasicDQN': BasicDQN,
        'CommutativeDQN': CommutativeDQN,
        'HallucinatedDQN': HallucinatedDQN
    }
    
    approaches = [approach_map[name](env, num_instances, seed, random_state, train_type, reward_type, noise_type) for name in approaches]
    learned_set = {approach.name: {problem_instance: None for problem_instance in problem_instances} for approach in approaches}
    
    for approach, problem_instance in itertools.product(approaches, problem_instances):
        learned_set[approach.name][problem_instance] = approach.get_adaptations(problem_instance)