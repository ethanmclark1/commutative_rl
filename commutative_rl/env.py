import os
import copy
import wandb
import pickle
import numpy as np
import networkx as nx
import gymnasium as gym
import problems.problem_generator as problems


class Env:
    def __init__(self, 
                 seed : int,
                 map_size: str, 
                 num_instances: int, 
                 reward_type: str,
                 noise_type: str
                 ) -> None:
        
        self.env = gym.make(
            id='FrozenLake-v1', 
            map_name=map_size, 
            render_mode=None
        )
        
        self.seed = seed
        self.reward_type = reward_type
        self.noise_type = noise_type
        
        self.action_rng = np.random.default_rng(seed)
        self.bridge_rng = np.random.default_rng(seed)
        self.instance_rng = np.random.default_rng(seed)
        
        self.num_cols = self.env.unwrapped.ncol if self.env.spec.kwargs['map_name'] == '8x8' else 5
        self.grid_dims = self.env.unwrapped.desc.shape if self.env.spec.kwargs['map_name'] == '8x8' else (5, 5)
        self.problem_size = self.env.spec.kwargs['map_name'] if self.env.spec.kwargs['map_name'] == '8x8' else '5x5'
        
        self.map_size = (self.num_cols, self.num_cols)
        
        self.state_dims = 64 if self.problem_size == '8x8' else 25
        self.max_elements = 12 if self.problem_size == '8x8' else 6
        self.action_cost = 0.025 if self.problem_size == '8x8' else 0.075
        self.action_dims = 16 + 1 if self.problem_size == '8x8' else 8 + 1
        
        self.name = self.__class__.__name__
        self.output_dir = f'earl/history/random_seed={self.seed}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        problems.generate_instances(self.problem_size, self.instance_rng, self.grid_dims, num_instances, self.state_dims)
        
        # Noise Parameters
        self.configs_to_consider = 1
        self.action_success_rate = 0.75
        self.percent_holes = 0.80 if noise_type == 'full' else 1
        
    def _save(self, problem_instance: str, adaptation: dict) -> None:
        directory = self.output_dir + f'/{self.name.lower()}'
        filename = f'{self.problem_size}_{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'wb') as file:
            pickle.dump(adaptation, file)
            
    def _load(self, problem_instance: str) -> dict:
        problem_instance = 'cheese'
        directory = self.output_dir + f'/{self.name.lower()}'
        filename = f'{self.problem_size}_{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            adaptation = pickle.load(f)
        return adaptation
    
    def _init_wandb(self, problem_instance: str) -> dict:
        if self.reward_type == 'true' and 'Commutative' in self.name:
            type_name = f'{self.name} w/ True Reward'
        elif self.reward_type == 'approximate':
            type_name = f'{self.name} w/ Approximate Reward'
        else:
            type_name = f'{self.name}'
        
        wandb.init(
            project=f'{self.problem_size} Frozen Lake', 
            entity='ethanmclark1', 
            name=f'{type_name}',
            tags=[f'{problem_instance.capitalize()}'],
            )
        
        config = wandb.config
        return config
    
    # Initialize action mapping for a given problem instance
    def init_instance(self, problem_instance: str) -> None:   
        self.instance = problems.get_instance(problem_instance)
        
    def generate_start_state(self) -> tuple:
        state = np.zeros(self.state_dims, dtype=int)
        bridges = []
        num_bridges = 0
        done = False
        
        return state, num_bridges, done
    
    def _place_bridge(self, state: np.ndarray, action: int) -> np.ndarray:
        next_state = copy.deepcopy(state)
        
        if action != 0:      
            next_state[action - 1] = 1
            
        return next_state
    
    def _reassign_states(self, 
                         prev_state: np.ndarray, 
                         prev_action: int, 
                         state: np.ndarray, 
                         action: int, 
                         next_state: np.ndarray
                         ) -> tuple:
        
        action_a_success = not np.array_equal(prev_state, state)
        action_b_success = not np.array_equal(state, next_state)
        
        commutative_state = self._place_bridge(prev_state, action)
    
        if action_a_success and action_b_success:         
            pass   
        elif not action_a_success and action_b_success:
            if prev_action != action:
                next_state = commutative_state
        elif action_a_success and not action_b_success:
            commutative_state = prev_state
            next_state = state
        else:
            commutative_state = prev_state
            
        return commutative_state, next_state
    
    def _create_instance(self, state: np.ndarray) -> tuple:
        graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
        nx.set_edge_attributes(graph, 1, 'weight')
        
        desc = copy.deepcopy(state).reshape(self.grid_dims)
        
        return graph, desc
        
    # Cell Values: {Frozen: 0, Bridge: 1, Start: 2, Goal: 3, Hole: 4}
    def _calc_utility(self, state: np.ndarray) -> float:
        def manhattan_dist(a: tuple, b: tuple) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        utilities = []
        graph, desc = self._create_instance(state)
        
        for _ in range(self.configs_to_consider):
            tmp_desc = copy.deepcopy(desc)
            tmp_graph = copy.deepcopy(graph)
            
            start, goal, holes = problems.get_entity_positions(self.instance, self.instance_rng, self.percent_holes)
            tmp_desc[start], tmp_desc[goal] = 2, 3
            
            # Only place holes if the cell is frozen
            for hole in holes:
                if tmp_desc[hole] == 0:
                    tmp_desc[hole] = 4
                    tmp_graph.remove_node(hole)
            
            path = nx.astar_path(tmp_graph, start, goal, manhattan_dist, 'weight')
            utility = -len(path)
            utilities.append(utility)

        return np.mean(utilities)
    
    def _get_next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        next_state = copy.deepcopy(state)
        
        if action != 0 and (self.action_success_rate == 1 or self.action_success_rate >= self.action_rng.random()):
            next_state = self._place_bridge(state, action)
            
        return next_state

    def _get_reward(self, state: np.ndarray, action: int, next_state: np.ndarray, num_action: int) -> tuple:
        reward = 0
        
        done = action == 0
        timeout = num_action == self.max_elements
        
        if not done:
            if not np.array_equal(state, next_state):
                util_s = self._calc_utility(state)
                util_s_prime = self._calc_utility(next_state)
                reward = util_s_prime - util_s
            reward -= self.action_cost * num_action
                        
        return reward, done or timeout
        
    def _step(self, state: np.ndarray, action: int, num_action: int) -> tuple:
        next_state = self._get_next_state(state, action)
        reward, done = self._get_reward(state, action, next_state, num_action)  
        
        return reward, next_state, done
    
    def get_adaptations(self, problem_instance: str) -> list:
        approach = self.__class__.__name__
        try:
            adaptations = self._load(problem_instance)
        except FileNotFoundError:
            print(f'No stored adaptation for {approach} on the {problem_instance.capitalize()} problem instance.')
            print('Generating new adaptation...')
            adaptations = self.generate_adaptations(problem_instance)
            self._save(problem_instance, adaptations)
        
        print(f'{approach} adaptations for {problem_instance.capitalize()} problem instance:\n{adaptations}\n')
        return adaptations
    
    def get_adapted_env(self, desc: np.ndarray, adaptations: list) -> np.ndarray:
        for bridge in adaptations:
            if hasattr(self, '_transform_action'):
                bridge = self._transform_action(bridge)
            row = bridge // self.num_cols
            col = bridge % self.num_cols
            if desc[row][col] == 2 or desc[row][col] == 3:
                continue
            desc[row][col] = 1
        
        return desc