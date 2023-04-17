import numpy as np

from shapely import Point
from agents.utils.base_aqent import BaseAgent

from language.ea import EA
from language.rl_agent import RLAgent
from language.grid_world import GridWorld

class Listener(BaseAgent):
    def __init__(self):
        super().__init__()
        self.actions = np.arange(1,5)
        
    # TODO: Implement Listener's constraints
    def _generate_constraints(self):
        a=3
            
    def get_action(self, observation, goal, directions, find_target, env):
        observation = observation[0:2]
        target = find_target(observation, goal, directions)
        
        obs_region = self.localize(observation)
        goal_region = self.localize(goal)
        if obs_region == goal_region:
            next_region = goal_region
            target = goal
        else:
            label = directions[directions.index(obs_region)+1:][0]
            next_region = self.language[label]
            target = next_region.centroid
        
        action = super().get_action(observation, target, env)
        return action
