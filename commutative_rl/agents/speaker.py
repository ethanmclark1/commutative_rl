import networkx as nx

from shapely import Point
from commutative_rl.env import Env

class Speaker:
    def __init__(self, num_agents: int, obstacle_radius: float) -> None:
        self.num_agents = num_agents
        self.obstacle_radius = obstacle_radius
    
    # Determine the positions of the agents, goals, and obstacles
    def gather_info(self, state) -> None:
        self.starts = []
        self.goals = []
        self.obstacles = None
        
        for idx in range(self.num_agents):
            self.starts += [state[2*idx : 2*idx+2]]
            self.goals += [state[2*self.num_agents + 2*idx : 2*self.num_agents + 2*idx + 2]]
            
        self.obstacles = state[4*self.num_agents : ].reshape(-1, 2)
    
    def direct(self, name: str, approach: object) -> list:
        if any(x in name for x in ['dqn', 'sac', 'voronoi_map']):
            obstacles = [Point(obstacle).buffer(self.obstacle_radius) for obstacle in self.obstacles]

            directions = []
            for start, goal in zip(self.starts, self.goals):
                graph, start_region, goal_region = Env.create_instance(approach, start, goal, obstacles)
                
                try:
                    directions += [nx.astar_path(graph, start_region, goal_region)]
                except (nx.NodeNotFound, nx.NetworkXNoPath):
                    directions += [None]        
        elif name == 'grid_world':
            directions = [approach.direct(start, goal, self.obstacles) for start, goal in zip(self.starts, self.goals)]
        else:
            directions = [approach]
        
        return directions