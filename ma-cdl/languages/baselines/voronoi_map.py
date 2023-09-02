import networkx as nx

from languages.utils.cdl import CDL
from shapely import Point, MultiPoint, Polygon, MultiPolygon, voronoi_polygons

class VoronoiMap:
    regions = None
    
    def __init__(self, show_animation=True):
        self.show_animation = show_animation
        self.bbox = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    
    # Create the voronoi map using Shapely
    def create_voronoi_map(self, start, goal, obstacles):
        points = MultiPoint([start, goal, *obstacles])
        voronoi_map = voronoi_polygons(points)
        return voronoi_map

    # Clip VoronoiMap to fit the bounding box
    def clip_voronoi(self, voronoi_map):
        clipped_voronoi = []
        for region in voronoi_map.geoms:
            clipped_polygon = self.bbox.intersection(region)
            if clipped_polygon.is_empty:
                continue
            elif clipped_polygon.geom_type == 'Polygon':
                clipped_voronoi.append(clipped_polygon)
            elif clipped_polygon.geom_type == 'MultiPolygon':
                clipped_voronoi.extend(clipped_polygon)
        return MultiPolygon(clipped_voronoi)
    
    def direct(self, start, goal, obstacles): 
        def euclidean_distance(a, b):
            return Point(VoronoiMap.regions[a].centroid).distance(Point(VoronoiMap.regions[b].centroid))
          
        voronoi_map = self.create_voronoi_map(start, goal, obstacles) 
        clipped_voronoi = self.clip_voronoi(voronoi_map) 
        VoronoiMap.regions = [*clipped_voronoi.geoms]
        
        agent_idx = CDL.localize(Point(start), VoronoiMap.regions)
        goal_idx = CDL.localize(Point(goal), VoronoiMap.regions)
        
        obstacles = [Point(obstacle) for obstacle in obstacles]
        
        safe_graph = CDL.get_safe_graph(VoronoiMap.regions, obstacles)
        
        try:
            directions = nx.astar_path(safe_graph, agent_idx, goal_idx, heuristic=euclidean_distance)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            directions = None

        return directions