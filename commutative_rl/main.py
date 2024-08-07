import signal8
import itertools

from arguments import get_arguments

from agents.speaker import Speaker
from agents.listener import Listener

from languages.baselines.grid_world import GridWorld
from languages.baselines.voronoi_map import VoronoiMap
from languages.baselines.direct_path import DirectPath
from commutative_rl.languages.discrete import BasicDQN, CommutativeDQN, HallucinatedDQN
from commutative_rl.languages.continuous import BasicSAC, CommutativeSAC


if __name__ == '__main__':
    num_agents, num_large_obstacles, num_small_obstacles, seed, approaches, problem_instances, reward_type, render_mode = get_arguments()
    
    env = signal8.env(
        num_agents=num_agents, 
        num_large_obstacles=num_large_obstacles, 
        num_small_obstacles=num_small_obstacles, 
        render_mode=render_mode,
        max_cycles=50
        )
    
    scenario = env.unwrapped.scenario
    world = env.unwrapped.world
    agent_radius = world.agents[0].radius
    obstacle_radius = world.small_obstacles[0].radius 
    
    approach_map = {
        'BasicDQN': BasicDQN,
        'CommutativeDQN': CommutativeDQN,
        'HallucinatedDQN': HallucinatedDQN,
        'BasicSAC': BasicSAC,
        'CommutativeSAC': CommutativeSAC,
    }
    
    approaches = [approach_map[name](scenario, world, seed, reward_type) for name in approaches]
    learned_set = {approach.name: {problem_instance: None for problem_instance in problem_instances} for approach in approaches}
        
    grid_world = GridWorld()
    voronoi_map = VoronoiMap(scenario, world, seed)
    direct_path = DirectPath(scenario, world, seed)
            
    aerial_agent = Speaker(num_agents, obstacle_radius)
    ground_agent = Listener(agent_radius, obstacle_radius)
    
    for approach, problem_instance in itertools.product(approaches, problem_instances):
        learned_set[approach.name][problem_instance] = approach.generate_language(problem_instance)