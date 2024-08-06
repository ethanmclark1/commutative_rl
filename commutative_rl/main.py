import copy
import signal8
import itertools
import numpy as np

from plotter import plot_metrics
from arguments import get_arguments

from agents.speaker import Speaker
from agents.listener import Listener

from languages.baselines.grid_world import GridWorld
from languages.baselines.voronoi_map import VoronoiMap
from languages.baselines.direct_path import DirectPath
from commutative_rl.languages.discrete import BasicDQN, CommutativeDQN, HallucinatedDQN
from commutative_rl.languages.continuous import BasicSAC, CommutativeSAC


class MA_CDL():
    def __init__(self, 
                 num_agents: int, 
                 seed: int,
                 num_large_obstacles: int, 
                 num_small_obstacles: int,
                 random_state: bool,
                 train_type: str,
                 reward_type: str,
                 render_mode: str
                 ) -> None:
        
        self.env = signal8.env(
            num_agents=num_agents, 
            num_large_obstacles=num_large_obstacles, 
            num_small_obstacles=num_small_obstacles, 
            render_mode=render_mode,
            max_cycles=50
            )
        
        scenario = self.env.unwrapped.scenario
        world = self.env.unwrapped.world
        agent_radius = world.agents[0].radius
        obstacle_radius = world.small_obstacles[0].radius 
        
        # Discrete RL
        self.basic_dqn = BasicDQN(scenario, world, seed, random_state, train_type, reward_type)
        self.commutative_dqn = CommutativeDQN(scenario, world, seed, random_state, train_type, reward_type)
        self.hallucinated_dqn = HallucinatedDQN(scenario, world, seed, random_state, train_type, reward_type)
        
        # Continuous RL
        self.basic_sac = BasicSAC(scenario, world, seed, random_state, train_type, reward_type)
        self.commutative_sac = CommutativeSAC(scenario, world, seed, random_state, train_type, reward_type)
                                        
        # Baselines
        self.grid_world = GridWorld()
        self.voronoi_map = VoronoiMap(scenario, world, seed)
        self.direct_path = DirectPath(scenario, world, seed)
                
        self.aerial_agent = Speaker(num_agents, obstacle_radius)
        self.ground_agent = Listener(agent_radius, obstacle_radius)
    
    def retrieve_language(self, name: str, problem: str) -> dict:     
        language_set = {}       
        
        approach = getattr(self, name)
        language_set[name] = approach.get_language(problem)
            
        return language_set
    
    def retrieve_baselines(self, language_set: dict, problem: str) -> dict:
        for name in ['grid_world', 'voronoi_map', 'direct_path']:
            approach = getattr(self, name)
            if hasattr(approach, 'get_language'):
                language_set[name] = approach.get_language(problem)
            else:
                language_set[name] = approach
                
        return language_set
        
    def evaluate(self, problem_instance, language_set, num_episodes):
        names = list(language_set.keys())
        direction_set = {name: None for name in names}

        language_safety = {name: 0 for name in names}
        ground_agent_success = {name: 0 for name in names}
        avg_direction_len = {name: 0 for name in names}
        direction_length = {name: [] for name in names}
        
        for _ in range(num_episodes):            
            self.env.reset(options={'problem_instance': problem_instance})
            start_state = self.env.state()
            self.aerial_agent.gather_info(start_state)
            
            # Create copy of world to reset to at beginning of each approach
            world = self.env.unwrapped.world
            backup = copy.deepcopy(world)
            
            direction_set = {name: self.aerial_agent.direct(name, language_set[name]) for name in names}
            
            for name, directions in direction_set.items(): 
                # Penalize if no directions are given
                if None in directions:
                    if any(x in name for x in ['dqn', 'sac', 'voronoi_map']):
                        directions = len(language_set[name])
                    elif name == 'grid_world':
                        directions = self.grid_world.graph.number_of_nodes()
                    else:
                        directions = 20
                    direction_length[name].append(directions)
                    continue
                
                language_safety[name] += 1
                max_directions = max(len(direction) for direction in directions)
                direction_length[name].append(max_directions)

                observation, _, termination, truncation, _ = self.env.last()
                while not (termination or truncation):
                    action = self.ground_agent.get_action(observation, directions, name, language_set[name])

                    # Epsisode terminates if ground agent doesn't adhere to directions
                    if action is not None:
                        self.env.step(action)
                        observation, _, termination, truncation, _ = self.env.last()
                    else:
                        truncation = True
                                
                    if termination:
                        ground_agent_success[name] += 1
                        self.env.terminations['agent_0'] = False
                        break
                    elif truncation:
                        self.env.truncations['agent_0'] = False
                        break   
                    
                self.env.unwrapped.steps = 0
                self.env.unwrapped.world = copy.deepcopy(backup)
        
        avg_direction_len = {name: np.mean(direction_length[name]) for name in names}

        return language_safety, ground_agent_success, avg_direction_len
        

if __name__ == '__main__':    
    num_agents, num_large_obstacles, num_small_obstacles, seed, names, problem_instances, reward_type, render_mode = get_arguments()
    ma_cdl = MA_CDL(num_agents, num_large_obstacles, num_small_obstacles, seed, reward_type, render_mode)

    all_metrics = []
    num_episodes = 1000
    for name, problem_instance in itertools.product(names, problem_instances):
        language_set = ma_cdl.retrieve_language(name, problem_instance)
    #     language_set = ma_cdl.retrieve_baselines(language_set, problem_instance)
    #     language_safety, ground_agent_success, avg_direction_len = ma_cdl.evaluate(problem_instance, language_set, num_episodes)

    #     all_metrics.append({
    #         'language_safety': language_safety,
    #         'ground_agent_success': ground_agent_success,
    #         'avg_direction_len': avg_direction_len,
    #     })

    # plot_metrics(problem_instances, all_metrics, num_episodes)