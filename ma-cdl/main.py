import copy
import cProfile
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from arguments import get_arguments

from agents.speaker import Speaker
from agents.listener import Listener

from environment import simple_path
from environment.utils.problems import problem_scenarios

from languages.td3 import TD3
from languages.evolutionary_algo import EA
from languages.bandit import Bandit, ContextualBandit

class MA_CDL2():
    def __init__(self, args):
        self.num_episodes = 10
        self.env = simple_path.env(args)
        
        obs_radius = self.env.metadata['obs_radius']
        num_obs = self.env.metadata['num_obstacles']
        agent_radius = self.env.metadata['agent_radius']
        
        self.ea = EA(agent_radius, obs_radius, num_obs)
        self.td3 = TD3(agent_radius, obs_radius, num_obs)
        self.bandit = Bandit(agent_radius, obs_radius, num_obs)
        self.contextual_bandit = ContextualBandit(agent_radius, obs_radius, num_obs)
        
        self.speaker = Speaker()
        self.listener = Listener()

    def act(self):
        approaches = ['ea', 'td3', 'bandit', 'contextual_bandit']
        directions = {approach: None for approach in approaches}
        languages = {approach: None for approach in approaches}
        direction_len = {approach: {scenario: [] for scenario in problem_scenarios} for approach in approaches}
        results = {approach: {scenario: 0 for scenario in problem_scenarios} for approach in approaches}
        
        for _, scenario in product(range(self.num_episodes), problem_scenarios):
            languages['td3'] = self.td3.get_language(scenario)
            languages['bandit'] = self.bandit.get_language(scenario)
            languages['contextual_bandit'] = self.contextual_bandit.get_language(scenario)
            
            self.env.reset(options={'problem_name': scenario})
            start, goal, obstacles = self.env.unwrapped.get_init_conditions()
            
            # TODO: Fix this
            directions['ea'] = self.speaker.direct(start, goal, obstacles, languages['ea'])    
            directions['td3'] = self.speaker.direct(start, goal, obstacles, languages['td3'])    
                                    
            world = self.env.unwrapped.world
            backup = copy.deepcopy(world)
            
            for approach in directions:                
                directions = directions[approach]
                if directions is None:
                    continue
                
                direction_len[approach][scenario].append(len(directions))
                self.env.unwrapped.world = copy.deepcopy(backup)
                obs, _, termination, truncation, _ = self.env.last()

                while not (termination or truncation):
                    action = self.listener.get_action(obs, goal, directions, self.env)
                    
                    # No action can adhere to the directions
                    if action is None:
                        break
                        
                    self.env.step(action)
                    obs, _, termination, truncation, _ = self.env.last()
                                
                    if termination:
                        results[approach][scenario] += 1
                        self.env.terminations['agent_0'] = False
                        break
                    elif truncation:
                        self.env.truncations['agent_0'] = False
                        break   
        
        avg_direction_len = {approach: {scenario: np.mean(values) for scenario, values in scenario_dict.items()} 
                         for approach, scenario_dict in direction_len.items()}
        return results, avg_direction_len
    
    def plot(self, results, avg_direction_len):        
        # Create the first figure for success rates
        fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
        fig1.suptitle('Success Rate Comparison between Context-Dependent Language, Voronoi Maps, and Grid World on Suite of Problem Scenarios')
        
        # Create the second figure for average direction lengths
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
        fig2.suptitle('Average Direction Length Comparison between Context-Dependent Language, Voronoi Maps, and Grid World on Suite of Problem Scenarios')
        
        num_iterations = len(axes1)
        num_labels = len(self.problem_scenarios)
        group_size = num_labels // num_iterations
        for i in range(num_iterations):
            start = i * group_size
            end = start + group_size if i < num_iterations - 1 else num_labels
            _labels = self.problem_scenarios[start:end]
            
            _cdl_results = list(results['language'].values())[start:end]
            _grid_world_results = list(results['grid_world'].values())[start:end]
            _voronoi_map_results = list(results['voronoi_map'].values())[start:end]

            # Plot the success rate graphs
            axes1[i].bar(np.arange(len(_labels)) - 0.2, _cdl_results, width=0.2, label='Context-Dependent Language')
            axes1[i].bar(np.arange(len(_labels)), _grid_world_results, width=0.2, label='Grid World')
            axes1[i].bar(np.arange(len(_labels)) + 0.2, _voronoi_map_results, width=0.2, label='Voronoi Maps')
            axes1[i].set_xlabel('Problem Type')
            axes1[i].set_ylabel('Success Rate (%)')
            axes1[i].set_xticks(np.arange(len(_labels)))
            axes1[i].set_xticklabels(_labels)
            axes1[i].set_ylim(0, 100)
            axes1[i].legend()

            _cdl_direction_len = list(avg_direction_len['language'].values())[start:end]
            _grid_world_direction_len = list(avg_direction_len['grid_world'].values())[start:end]
            _voronoi_map_direction_len = list(avg_direction_len['voronoi_map'].values())[start:end]
            
            # Plot the average direction length graphs
            axes2[i].bar(np.arange(len(_labels)) - 0.2, _cdl_direction_len, width=0.2, label='Context-Dependent Language')
            axes2[i].bar(np.arange(len(_labels)), _grid_world_direction_len, width=0.2, label='Grid World')
            axes2[i].bar(np.arange(len(_labels)) + 0.2, _voronoi_map_direction_len, width=0.2, label='Voronoi Maps')
            axes2[i].set_xlabel('Problem Type')
            axes2[i].set_ylabel('Average Direction Length')
            axes2[i].set_xticks(np.arange(len(_labels)))
            axes2[i].set_xticklabels(_labels)
            axes2[i].legend()
        
        # Save the figures
        fig1.savefig('success_rates.png')
        fig2.savefig('average_direction_lengths.png')
        
        # Show the plots in separate windows
        plt.show()

if __name__ == '__main__':
    args = get_arguments()
    ma_cdl2 = MA_CDL2(args)
    results, avg_direction_len = ma_cdl2.act()
    ma_cdl2.plot(results, avg_direction_len)