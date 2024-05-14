import numpy as np
import geopandas as gpd

from sklearn.cluster import DBSCAN
from languages.utils.cdl import CDL
from longsgis import voronoiDiagram4plg
from shapely import MultiPoint, Polygon, MultiPolygon, box


class VoronoiMap():    
    def __init__(self, scenario: object, world: object, seed: int, show_animation=True) -> None:  
        self.scenario = scenario
        self.world = world      
        self.seed = seed
        self.num_configs = 100
        self.show_animation = show_animation
        
        self.box = box(-1, -1, 1, 1)
        self.bounding_polygon = Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
        
        self.world_rng = np.random.default_rng(seed=seed)
        
    def get_language(self, problem_instance) -> list:
        obstacles = []
        for _ in range(self.num_configs):
            _, _, obs = CDL.get_entity_positions(self.scenario, self.world, self.world_rng, problem_instance)
            obstacles.extend(obs)
        
        # Cluster obstacles into k clusters to uncover their constraints
        dbscan = DBSCAN(eps=0.3, min_samples=5).fit(obstacles)
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # -1 label is for noise
        
        clusters = {i: [] for i in range(n_clusters)}
        for label, obstacle in zip(labels, obstacles):
            if label != -1:  # Ignore noise
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