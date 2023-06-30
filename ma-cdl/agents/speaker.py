import networkx as nx

from shapely import Point
from languages.utils.cdl import CDL
from agents.utils.a_star import a_star

class Speaker:
    def __init__(self, num_agents, obstacle_radius):
        self.agents = []
        self.goals = []
        self.obstacles = None
        self.num_agents = num_agents
        self.obstacle_radius = obstacle_radius
        self.languages = ['EA', 'TD3', 'Bandit']
    
    # Determine the positions of the agents, goals, and obstacles
    def gather_info(self, state):
        for idx in range(self.num_agents):
            self.agents += [state[idx*2 : idx*2+2]]
            self.goals += [state[self.num_agents*2 + idx*2 : self.num_agents*2 + idx*2 + 2]]
            
        self.obstacles = state[self.num_agents*4 : ].reshape(-1, 2)
    
    # Get the directions for each agent
    def direct(self, approach):
        if isinstance(approach, list):
            directions = self.direct_with_cdl(approach)
        else:
            directions = self.direct_with_baseline(approach)
        
        return directions
            
    def direct_with_cdl(self, approach):
        directions = []
        
        obstacles = [Point(obstacle).buffer(self.obstacle_radius) 
                     for obstacle in self.obstacles]
    
        for agent, goal in zip(self.agents, self.goals):
            agent_idx = CDL.localize(agent, approach)
            goal_idx = CDL.localize(goal, approach)
            
            try:
                directions += [a_star(agent_idx, goal_idx, obstacles, approach)]
            except TypeError:
                graph = CDL.create_graph(approach)
                directions += [nx.shortest_path(graph, agent_idx, goal_idx)]
        
    def direct_with_baseline(self, approach):
        directions = []
        
        for agent, goal in zip(self.agents, self.goals):
            directions += [approach.direct(agent, goal, self.obstacles)]
            
        return directions