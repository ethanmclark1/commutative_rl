import torch
import numpy as np


def encode(
    input_value: int | list, max_val: int, to_tensor: bool = False
) -> float | list:
    if isinstance(input_value, list):
        encoded = [val / max_val for val in input_value]
    else:
        encoded = input_value / max_val

    if to_tensor:
        encoded = torch.as_tensor(encoded, dtype=torch.float32).view(-1)

    return encoded


import itertools
import networkx as nx

from shapely import LineString, MultiLineString, Polygon, Point, box

CORNERS = list(itertools.product((1, -1), repeat=2))
BOUNDARIES = [
    LineString([CORNERS[0], CORNERS[2]]),
    LineString([CORNERS[2], CORNERS[3]]),
    LineString([CORNERS[3], CORNERS[1]]),
    LineString([CORNERS[1], CORNERS[0]]),
]
SQUARE = Polygon([CORNERS[2], CORNERS[0], CORNERS[1], CORNERS[3]])
BOX = box(-1, -1, 1, 1)


def convert_to_linestring(lines: tuple) -> list:
    linestrings = []
    lines = np.reshape(lines, (-1, 3))
    for line in lines:
        a, b, c = line

        if a == 0 and b == 0 and c == 0:  # Terminal line
            break
        elif a == 0:  # Horizontal line
            start, end = (-1, -c / b), (1, -c / b)
        elif b == 0:  # Vertical line
            start, end = (-c / a, -1), (-c / a, 1)
        else:
            slope = a / -b
            if abs(slope) >= 1:
                y1 = (-a + c) / -b
                y2 = (a + c) / -b
                start, end = (-1, y1), (1, y2)
            else:
                x1 = (-b + c) / -a
                x2 = (b + c) / -a
                start, end = (x1, -1), (x2, 1)

        linestrings.append(LineString([start, end]))

    return linestrings


def get_intersecting_lines(linestrings: list) -> list:
    valid_lines = list(BOUNDARIES)

    for linestring in linestrings:
        intersection = SQUARE.intersection(linestring)
        if not intersection.is_empty and not intersection.geom_type == "Point":
            coords = np.array(intersection.coords)
            if np.any(np.abs(coords) == 1, axis=1).all():
                valid_lines.append(intersection)

    return valid_lines


"""WARNING: Changing this distance requires that distance in the safe_graph function be changed"""


def create_regions(valid_lines: list, distance: float = 2e-4) -> list:
    lines = MultiLineString(valid_lines).buffer(distance=distance)
    boundary = lines.convex_hull
    polygons = boundary.difference(lines)
    regions = [polygons] if polygons.geom_type == "Polygon" else list(polygons.geoms)
    return regions


def localize(entity: np.ndarray, language: list) -> int:
    point = Point(entity)
    region_idx = next(
        (idx for idx, region in enumerate(language) if region.contains(point)), None
    )
    return region_idx


def get_entity_positions(
    scenario: object,
    world: object,
    world_rng: np.random.default_rng,
    problem_instance: str,
) -> tuple:
    scenario.reset_world(world, world_rng, problem_instance)

    rand_idx = world_rng.choice(len(world.agents))
    start = world.agents[rand_idx].state.p_pos
    goal = world.agents[rand_idx].goal.state.p_pos
    obstacles = [obs.state.p_pos for obs in world.large_obstacles]

    return start, goal, obstacles


# Create graph from language excluding regions with obstacles
def create_instance(
    regions: list, start: np.ndarray, goal: np.ndarray, obstacles: list
) -> tuple:
    graph = nx.Graph()

    obstacle_regions = {
        idx
        for idx, region in enumerate(regions)
        if any(region.intersects(obstacle) for obstacle in obstacles)
    }

    # Add nodes to graph
    for idx, region in enumerate(regions):
        if idx in obstacle_regions:
            continue
        centroid = region.centroid
        graph.add_node(idx, position=(centroid.x, centroid.y))

    for idx, region in enumerate(regions):
        if idx in obstacle_regions:
            continue

        for neighbor_idx, neighbor in enumerate(regions):
            if idx == neighbor_idx or neighbor_idx in obstacle_regions:
                continue

            if region.dwithin(neighbor, 4.0000001e-4):
                graph.add_edge(idx, neighbor_idx)

    start_region = localize(start, regions)
    goal_region = localize(goal, regions)

    return graph, start_region, goal_region
