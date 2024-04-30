import numpy as np

from scipy.spatial import cKDTree
from languages.utils.cdl import CDL
from languages.baselines.utils.rrt_star import RRTStar


class DirectPath:
    tree = None
    
    def __init__(self, scenario, world, seed):
        self.num_configs = 2000
        self.scenario = scenario
        self.world = world
        self.world_rng = np.random.default_rng(seed=seed)
        
        agent_radius = world.agents[0].radius
        goal_radius = world.goals[0].radius
        obstacle_radius = world.small_obstacles[0].radius 
        self.planner = RRTStar(agent_radius, goal_radius, obstacle_radius)
    
    @staticmethod
    def get_point_index(point):
        return DirectPath.tree.query(point)[1]
    
    def get_language(self, problem_instance):
        obstacles = []
        direct_path = None
        for _ in range(self.num_configs):
            start, goal, obs = CDL.get_entity_positions(self.scenario, self.world, self.world_rng, problem_instance)
            obstacles.extend(obs)
            
        while direct_path is None:
            direct_path = self.planner.get_path(start, goal, obstacles)
        DirectPath.tree = cKDTree(direct_path)
        
        return direct_path