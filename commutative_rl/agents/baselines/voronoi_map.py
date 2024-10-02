import numpy as np
import geopandas as gpd

from env import Env
from agents.utils.helpers import *
from sklearn.cluster import DBSCAN
from longsgis import voronoiDiagram4plg
from shapely import MultiPoint, MultiPolygon


class VoronoiMap:
    def __init__(
        self,
        seed: int,
        num_agents: int,
        num_large_obstacles: int,
        num_small_obstacles: int,
        config: dict,
    ) -> None:

        self.name = self.__class__.__name__

        self.env = Env(
            seed,
            num_agents,
            num_large_obstacles,
            num_small_obstacles,
            config["env"],
        )

        self.world_rng = np.random.default_rng(seed)

    def generate_language(self, problem_instance) -> list:
        obstacles = []
        for _ in range(self.num_configs):
            _, _, obs = get_entity_positions(
                self.env.scenario, self.env.world, self.world_rng, problem_instance
            )
            obstacles.extend(obs)

        dbscan = DBSCAN(eps=0.3, min_samples=5).fit(obstacles)
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (
            1 if -1 in labels else 0
        )  # -1 label is for noise

        clusters = {i: [] for i in range(n_clusters)}
        for label, obstacle in zip(labels, obstacles):
            if label != -1:  # Ignore noise
                clusters[label].append(obstacle)
        cluster_shapes = [
            MultiPoint(cluster).convex_hull for cluster in clusters.values()
        ]
        polygon_with_holes = SQUARE.difference(MultiPolygon(cluster_shapes))

        polygons = [polygon_with_holes.exterior] + list(polygon_with_holes.interiors)
        polygons_gdf = gpd.GeoDataFrame(geometry=polygons)
        polygons_gdf.crs = 32650

        boundary_gdf = gpd.GeoDataFrame(geometry=[BOX])
        boundary_gdf.crs = 32650

        voronoi_diagram = voronoiDiagram4plg(polygons_gdf, boundary_gdf)
        return [*voronoi_diagram.geometry]
