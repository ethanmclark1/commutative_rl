import numpy as np
import geopandas as gpd

from sklearn.cluster import KMeans
from languages.utils.cdl import CDL
from longsgis import voronoiDiagram4plg

from shapely import MultiPoint, Polygon, MultiPolygon, box


class VoronoiMap():    
    def __init__(self, scenario: object, world: object, seed: int, show_animation=True):  
        self.scenario = scenario
        self.world = world      
        self.num_configs = 2000
        self.show_animation = show_animation
        self.box = box(-1, -1, 1, 1)
        self.bounding_polygon = Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
        
        self.world_rng = np.random.default_rng(seed=seed)
        
        self.num_k = {
            'bisect': 1, 
            'circle': 1, 
            'cross': 1, 
            'corners': 4, 
            'staggered': 4,
            'quarters': 4, 
            'scatter': 4, 
            'stellaris': 5
            }
        
    def get_language(self, problem_instance):
        obstacles = []
        for _ in range(self.num_configs):
            _, _, obs = CDL.get_entity_positions(self.scenario, self.world, self.world_rng, problem_instance)
            obstacles.extend(obs)
        
        # Cluster obstacles into k clusters to uncover their constraints
        k_means = KMeans(n_clusters=self.num_k[problem_instance], random_state=0, n_init=10).fit(obstacles)
        labels = k_means.labels_
        clusters = {i: [] for i in range(self.num_k[problem_instance])}
        for label, obstacle in zip(labels, obstacles):
            clusters[label].append(obstacle)
        cluster_shapes = [MultiPoint(cluster).convex_hull for cluster in clusters.values()]
        polygon_with_holes = self.bounding_polygon.difference(MultiPolygon(cluster_shapes))
        
        polygons = [polygon_with_holes.exterior] + list(polygon_with_holes.interiors)
        polygons_gdf = gpd.GeoDataFrame(geometry=polygons)
        polygons_gdf.crs = 32650

        boundary_gdf = gpd.GeoDataFrame(geometry=[self.box])
        boundary_gdf.crs = 32650

        voronoi_diagram = voronoiDiagram4plg(polygons_gdf, boundary_gdf)

        return [*voronoi_diagram.geometry]