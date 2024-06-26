import io
import os
import torch
import wandb
import pickle
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from PIL import Image
from statistics import mean
from itertools import product
from shapely.geometry import Point, LineString, MultiLineString, Polygon

warnings.filterwarnings('ignore', message='invalid value encountered in intersection')

CORNERS = list(product((1, -1), repeat=2))
BOUNDARIES = [LineString([CORNERS[0], CORNERS[2]]),
              LineString([CORNERS[2], CORNERS[3]]),
              LineString([CORNERS[3], CORNERS[1]]),
              LineString([CORNERS[1], CORNERS[0]])]
SQUARE = Polygon([CORNERS[2], CORNERS[0], CORNERS[1], CORNERS[3]])

class CDL:
    def __init__(self, 
                 scenario: object,
                 world: object,
                 seed: int,
                 random_state: bool,
                 train_type: str,
                 reward_type: str
                 ) -> None:
        
        self.world = world
        self.scenario = scenario
        self.seed = seed
        self.random_state = random_state
        self.train_type = train_type
        self.reward_type = reward_type
        
        self.state_rng = np.random.default_rng(seed)
        self.world_rng = np.random.default_rng(seed)
        
        self.max_action = 3
        self.action_cost = 0.1
        self.util_multiplier = 1
        self.failed_path_cost = 0
        
        self.valid_lines = set()
        self._generate_start_state = self._generate_random_state if random_state else self._generate_fixed_state
        
        # Noise Parameters
        self.configs_to_consider = 100
        self.obstacle_radius = world.large_obstacles[0].radius
        
        self.name = self.__class__.__name__
        env_type = 'discrete' if 'DQN' in self.name else 'continuous'
        self.output_dir = f'ma-cdl/history/{env_type}/numobs{len(world.large_obstacles)}_cost{self.action_cost}_mult{self.util_multiplier}_failed{self.failed_path_cost}_configs{self.configs_to_consider}_seed{self.seed}'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _save(self, problem_instance: str, language: list) -> None:
        directory = self.output_dir + f'/{self.name.lower()}'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb') as file:
            pickle.dump(language, file)
    
    def _load(self, problem_instance: str) -> list:
        problem_instance = 'cheese'
        directory = self.output_dir + f'/{self.name.lower()}'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            language = pickle.load(f)
        return language
    
    def _init_wandb(self, problem_instance: str) -> dict:
        if self.reward_type == 'true' and 'Commutative' in self.name:
            type_name = f'{self.name} w/ True Reward'
        elif self.reward_type == 'approximate':
            type_name = f'{self.name} w/ Approximate Reward'
        else:
            type_name = f'{self.name}'
        
        wandb.init(
            project=f'{self.__class__.__name__[-3:]}', 
            entity='ethanmclark1', 
            name=f'{type_name}',
            tags=[f'{problem_instance.capitalize()}']
            )
        
        config = wandb.config
        return config
    
    # Visualize regions that define the language
    def _visualize(self, problem_instance: str, language: list) -> None:
        plt.clf()
        plt.cla()
        
        for idx, region in enumerate(language):
            plt.fill(*region.exterior.xy)
            plt.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')

        directory = self.output_dir + f'/{self.name.lower()}'
        filename = f'{problem_instance}.png'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(file_path)
        plt.cla()
        plt.clf()
        plt.close('all')
        
    # Log image of partitioned regions to Weights & Biases
    def _log_regions(self, problem_instance: str, title_name: str, title_data: int, regions: list, reward: int) -> None:
        _, ax = plt.subplots()
        problem_instance = problem_instance.capitalize()
        for idx, region in enumerate(regions):
            ax.fill(*region.exterior.xy)
            ax.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        ax.set_title(f'Problem Instance: {problem_instance}   {title_name.capitalize()}: {title_data}   Reward: {reward:.2f}')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        pil_image = Image.open(buffer)
        plt.close()
        wandb.log({"image": wandb.Image(pil_image)})
        
    def _generate_fixed_state(self) -> tuple:
        empty_action = [0] if 'DQN' in self.name else np.array([0.] * 3)
        
        state = self.max_action * empty_action
        regions = [SQUARE]
        adaptations = []
        num_action = len(adaptations)
        done = False
        
        return state, regions, adaptations, num_action, done
        
    def _generate_random_state(self) -> tuple:
        num_actions = self.state_rng.choice(self.max_action)
        
        if hasattr(self, 'candidate_lines'):
            adaptations = self.state_rng.choice(len(self.candidate_lines), size=num_actions, replace=False)
            actions = np.array(self.candidate_lines)[adaptations]
        else:
            actions = adaptations = self.state_rng.uniform(size=(num_actions, 3))        
               
        linestrings = CDL.get_shapely_linestring(actions)
        valid_lines = CDL.get_valid_lines(linestrings)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        
        if 'DQN' in self.name:
            empty_action = [0]
            state = sorted(list(adaptations)) + (self.max_action - len(adaptations)) * empty_action
        else:
            empty_action = np.array([0.] * 3)
            state = np.concatenate(sorted(list(actions), key=np.sum) + (self.max_action - len(actions)) * [empty_action])
        
        num_action = len(adaptations)
        done = False
        
        return state, regions, adaptations, num_action, done
            
    # Generate shapely linestring (startpoint & endpoint) from standard form of line (Ax + By + C = 0)
    @staticmethod
    def get_shapely_linestring(lines: tuple) -> list:
        linestrings = []
        lines = np.reshape(lines, (-1, 3))
        for line in lines:
            a, b, c = line
            
            if a == 0 and b == 0 and c == 0: # Terminal line
                break
            elif a == 0:  # Horizontal line
                start, end = (-1, -c/b), (1, -c/b)
            elif b == 0:  # Vertical line
                start, end = (-c/a, -1), (-c/a, 1)
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

    # Find the intersections between lines and the environment boundary
    @staticmethod
    def get_valid_lines(linestrings: list) -> list:
        valid_lines = list(BOUNDARIES)

        for linestring in linestrings:
            intersection = SQUARE.intersection(linestring)
            if not intersection.is_empty and not intersection.geom_type == 'Point':
                coords = np.array(intersection.coords)
                if np.any(np.abs(coords) == 1, axis=1).all():
                    valid_lines.append(intersection)

        return valid_lines    
    
    # Create polygonal regions from lines
    """WARNING: Changing this distance requires that distance in the safe_graph function be changed"""
    @staticmethod
    def create_regions(valid_lines: list, distance: float=2e-4) -> list:
        lines = MultiLineString(valid_lines).buffer(distance=distance)
        boundary = lines.convex_hull
        polygons = boundary.difference(lines)
        regions = [polygons] if polygons.geom_type == 'Polygon' else list(polygons.geoms)
        return regions 
    
    @staticmethod
    def localize(entity: np.ndarray, language: list) -> int:
        point = Point(entity)
        region_idx = next((idx for idx, region in enumerate(language) if region.contains(point)), None)
        return region_idx
    
    @staticmethod
    def get_entity_positions(scenario: object, world: object, world_rng: np.random.default_rng, problem_instance: str) -> tuple:
        scenario.reset_world(world, world_rng, problem_instance)
        
        rand_idx = world_rng.choice(len(world.agents))
        start = world.agents[rand_idx].state.p_pos
        goal = world.agents[rand_idx].goal.state.p_pos
        obstacles = [obs.state.p_pos for obs in world.large_obstacles]

        return start, goal, obstacles
    
    @staticmethod
    def debug(regions: list, start: np.ndarray, goal: np.ndarray, obstacles_with_size: list):
        for idx, region in enumerate(regions):
            plt.fill(*region.exterior.xy)
            plt.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        
        plt.plot(start[0], start[1], 'go')
        plt.plot(goal[0], goal[1], 'ro')
        [plt.fill(*obstacle.exterior.xy, color='yellow') for obstacle in obstacles_with_size]
        plt.show()
        
    # Create graph from language excluding regions with obstacles
    @staticmethod
    def create_instance(regions: list, start: np.ndarray, goal: np.ndarray, obstacles: list) -> tuple:
        graph = nx.Graph()

        obstacle_regions = {idx for idx, region in enumerate(regions) if any(region.intersects(obstacle) for obstacle in obstacles)}
        
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
                    
        start_region = CDL.localize(start, regions)
        goal_region = CDL.localize(goal, regions)

        return graph, start_region, goal_region
    
    def _calc_utility(self, problem_instance: str, regions: list) -> float:    
        def euclidean_dist(a: int, b: int) -> float:
            return regions[a].centroid.distance(regions[b].centroid)
              
        utilities = []
        cheese = False
        for _ in range(self.configs_to_consider):
            start, goal, obstacles = CDL.get_entity_positions(self.scenario, self.world, self.world_rng, problem_instance)
            obstacles_with_size = [Point(obs_pos).buffer(self.obstacle_radius) for obs_pos in obstacles]
            graph, start_region, goal_region = CDL.create_instance(regions, start, goal, obstacles_with_size)
            
            if cheese:
                CDL.debug(regions, start, goal, obstacles_with_size)
            
            try:
                path = nx.astar_path(graph, start_region, goal_region, euclidean_dist)
                safe_area = [regions[idx].area for idx in path]
                utility = self.util_multiplier * mean(safe_area)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                utility = self.failed_path_cost
            
            utilities.append(utility)

        return np.mean(utilities)
    
    def _get_regions(self, state: list) -> list:    
        if hasattr(self, 'candidate_lines'):
            _state = [int(i) for i in state]
            _state = [self.candidate_lines[i] for i in _state]
        else:
            _state = state
                
        linestring = CDL.get_shapely_linestring(_state)
        valid_lines = CDL.get_valid_lines(linestring)
        next_regions = CDL.create_regions(valid_lines)
        
        return next_regions
    
    # Append action to state and sort, then get next regions
    def _get_next_state(self, state: list, action: int) -> tuple: 
        was_tensor = True
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float) 
            was_tensor = False
        
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor([action])

        next_state = state.clone()
        _action = action.clone()

        single_sample = len(next_state.shape) == 1
        
        if 'DQN' in self.name:
            next_state[next_state == 0] = self.action_dims + 1
            next_state = torch.cat([next_state, _action], dim=-1)
            next_state = torch.sort(next_state, dim=-1).values
            next_state[next_state == self.action_dims + 1] = 0
            
            next_state = next_state[:-1] if single_sample else next_state[:, :-1]
            
            if not was_tensor:
                next_state = next_state.tolist()
        else:
            next_state[next_state == 0] = 1.
            
            # Ensure next_state and action are at least 2D; reshape if 1D
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)
            if _action.dim() == 1:
                _action = _action.unsqueeze(0)

            # Reshape to have the size [-1, N, 3] where N is dynamic
            next_state = next_state.reshape(-1, next_state.shape[-1] // 3, 3)
            _action = _action.reshape(-1, _action.shape[-1] // 3, 3)

            next_state = torch.cat([next_state, _action], dim=1)
            row_sums = torch.sum(next_state, dim=-1, keepdim=True)
            sorted_indices = torch.argsort(row_sums, dim=1)
            # Expand sorted_indices to match next_state dimensions for gathering
            sorted_indices = sorted_indices.expand(-1, -1, next_state.shape[-1])
            next_state = torch.gather(next_state, 1, sorted_indices)
            next_state[next_state == 1.] = 0

            # Remove the last element in the middle dimension and flatten if original was 1D
            if single_sample:
                next_state = next_state.squeeze(0)[:-1].flatten()
            else:
                next_state = next_state[:, :-1].reshape(next_state.shape[0], -1)
                
            if not was_tensor:
                next_state = next_state.numpy()
                              
        return next_state
    
    # r(s,a,s') = u(s') - u(s) - c(a)
    def _get_reward(self, problem_instance: str, regions: list, action: int, next_regions: list, num_action: int) -> tuple:        
        reward = 0
        
        if hasattr(self, 'candidate_lines'):
            done = action == 0
        else:
            done = self._is_terminating_action(action)
        timeout = num_action == self.max_action
        
        if not done:
            if len(regions) != len(next_regions):
                util_s = self._calc_utility(problem_instance, regions)
                util_s_prime = self._calc_utility(problem_instance, next_regions)
                reward = util_s_prime - util_s
            reward -= self.action_cost * num_action
            
        return reward, done or timeout
            
    # Overlay line in the environment
    def _step(self, problem_instance: str, state: list, regions: list, action, num_action: int) -> tuple: 
        if action == 0:
            next_state = state
            next_regions = regions
        else:
            next_state = self._get_next_state(state, action)
            next_regions = self._get_regions(next_state)
            
        reward, done = self._get_reward(problem_instance, regions, action, next_regions, num_action)
        
        if done:
            self.valid_lines.clear()
            
        return reward, next_state, next_regions, done
    
    def get_language(self, problem_instance: str) -> list:
        try:
            language = self._load(problem_instance)
        except FileNotFoundError:
            approach = self.__class__.__name__
            print(f'No stored language for {approach} on the {problem_instance.capitalize()} problem instance.')
            print('Generating new language...\n')
            language = self._generate_language(problem_instance)
            linestrings = CDL.get_shapely_linestring(language)
            valid_lines = CDL.get_valid_lines(linestrings)
            language = CDL.create_regions(valid_lines)
            self._visualize(problem_instance, language)
            self._save(problem_instance, language)
        
        return language