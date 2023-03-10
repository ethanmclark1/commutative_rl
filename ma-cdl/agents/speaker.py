import copy
import numpy as np

from math import inf
from itertools import product
from shapely import points, Polygon
from agents.base_aqent import BaseAgent

class Speaker(BaseAgent):
    def __init__(self):
        super().__init__()
    
    # Get info on neighboring regions
    def _find_neighbors(self, cur_region, goal_region, obstacles):
        neighbors = {}
        cur_region = self.language[cur_region]
        goal_region = self.language[goal_region]
        for neighbor in self.language:
            if not cur_region.equals_exact(neighbor, 0) and cur_region.dwithin(neighbor, 1e-10):
                idx = self.language.index(neighbor)
                g = cur_region.centroid.distance(neighbor.centroid)
                h = neighbor.centroid.distance(goal_region.centroid)
                f = g + h
                is_goal = neighbor.equals(goal_region)
                is_safe = not any(neighbor.contains(obstacles))
                neighbors[idx] = (is_goal, is_safe, f)
                
        return neighbors
    
    # Find optimal next region
    def _get_next_region(self, prev_region, cur_region, goal_region, obstacles):
        min_f = inf
        next_region = None
        obstacles = points(obstacles)
        neighbors = self._find_neighbors(cur_region, goal_region, obstacles)
        
        for idx, neighbor in neighbors.items():
            if neighbor[0]:
                next_region = idx
                break
            elif neighbor[1] and neighbor[2] < min_f:
                min_f = neighbor[2]
                next_region = idx
        
        if next_region is None or prev_region == next_region:
            a=3

        return next_region
        
    # Gather directions based on path and obstacles
    def direct(self, start, goal, obstacles):
        directions = []
        prev_region = None
        start, goal = points(start, goal)
        cur_region = self.localize(start)
        goal_region = self.localize(goal)
        directions.append(cur_region)
        
        while cur_region != goal_region:
            next_region = self._get_next_region(prev_region, cur_region, goal_region, obstacles)
            directions.append(next_region)
            prev_region = cur_region
            cur_region = next_region
            
        return directions
    

        
        
