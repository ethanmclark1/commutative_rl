import itertools
import numpy as np
import networkx as nx

from shapely import LineString, MultiLineString, Polygon, Point, box

CORNERS = list(itertools.product((1, -1), repeat=2))
BOUNDARY_LINES = [
    LineString([CORNERS[0], CORNERS[2]]),
    LineString([CORNERS[2], CORNERS[3]]),
    LineString([CORNERS[3], CORNERS[1]]),
    LineString([CORNERS[1], CORNERS[0]]),
]
SQUARE = Polygon([CORNERS[2], CORNERS[0], CORNERS[1], CORNERS[3]])
BOX = box(-1, -1, 1, 1)


def random_num_in_range(rng: np.random.default_rng, low: float, high: float) -> float:
    random_val = rng.random()
    val_in_range = random_val * (high - low) + low
    return val_in_range


# Denormalize line indexes to their original values if they're not -1 (i.e. uninitialized)
def denormalize(line_idxs: list, max_idx: int) -> list:
    return [int(line_idx * max_idx) for line_idx in line_idxs if line_idx != -1]


def convert_to_linestring(lines: tuple) -> list:
    linestrings = []
    lines = np.reshape(lines, (-1, 3))

    for line in lines:
        a, b, c = line

        if a == 0:  # Horizontal line
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
    valid_lines = BOUNDARY_LINES.copy()

    for linestring in linestrings:
        intersection = SQUARE.intersection(linestring)
        if not intersection.is_empty and not intersection.geom_type == "Point":
            coords = np.array(intersection.coords)
            if np.any(np.abs(coords) == 1, axis=1).all():
                valid_lines.append(intersection)

    return valid_lines


# WARNING: Changing distance here requires changing distance in create_instance function as well
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

    # rand_idx is for future work when there are multiple agents
    rand_idx = world_rng.choice(len(world.agents))
    start = world.agents[rand_idx].state.p_pos
    goal = world.agents[rand_idx].goal.state.p_pos
    # Generate obstacles as circles with radius equal to radius of large obstacles
    obstacle_radius = world.large_obstacles[0].radius
    obstacles = [
        Point(obs.state.p_pos).buffer(obstacle_radius) for obs in world.large_obstacles
    ]

    return start, goal, obstacles


def create_instance(
    regions: list,
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list,
    distance: float = 4.0000001e-4,
) -> tuple:

    # Build an empty graph
    graph = nx.Graph()

    # Use .intersects() instead of .contains() as it's less strict
    regions_with_obstacles = {
        region_idx
        for region_idx, region in enumerate(regions)
        if any(region.intersects(obstacle) for obstacle in obstacles)
    }

    # Add regions without obstacles as nodes in graph
    for region_idx, region in enumerate(regions):
        if region_idx in regions_with_obstacles:
            continue

        graph.add_node(region_idx, position=(region.centroid.x, region.centroid.y))

    # Add edges between valid regions that are within a certain distance
    for region_idx, region in enumerate(regions):
        if region_idx in regions_with_obstacles:
            continue

        for neighbor_idx, neighbor in enumerate(regions):
            if region_idx == neighbor_idx or neighbor_idx in regions_with_obstacles:
                continue

            if region.dwithin(neighbor, distance):
                graph.add_edge(region_idx, neighbor_idx)

    # Localize start and goal entities
    start_region = localize(start, regions)
    goal_region = localize(goal, regions)

    return graph, start_region, goal_region
