import os
import copy
import wandb
import pickle
import problems
import numpy as np
import networkx as nx


class Env:
    def __init__(self, 
                 env: object, 
                 total_num_instances: int, 
                 seed: int, 
                 random_state: bool, 
                 train_type: str,
                 reward_type: str,
                 noise_type: str
                 ) -> None:
        
        self.env = env
        self.seed = seed
        self.random_state = random_state
        self.train_type = train_type
        self.reward_type = reward_type
        self.noise_type = noise_type
        
        self.action_rng = np.random.default_rng(seed)
        self.bridge_rng = np.random.default_rng(seed)
        self.instance_rng = np.random.default_rng(seed)
        
        self.num_cols = env.unwrapped.ncol if env.spec.kwargs['map_name'] == '8x8' else 5
        self.grid_dims = env.unwrapped.desc.shape if env.spec.kwargs['map_name'] == '8x8' else (5, 5)
        self.problem_size = env.spec.kwargs['map_name'] if env.spec.kwargs['map_name'] == '8x8' else '5x5'
        
        self.max_action = 12 if self.problem_size == '8x8' else 6
        self.state_dims = 16 if self.problem_size == '8x8' else 8
        self.action_cost = 0.025 if self.problem_size == '8x8' else 0.075
        # Add a dummy action (+1) to terminate the episode
        self.action_dims = self.state_dims + 1  
        
        self.name = self.__class__.__name__
        self.output_dir = f'earl/history/random_seed={self.seed}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        problems.generate_instances(self.problem_size, self.instance_rng, self.grid_dims, total_num_instances, self.state_dims)
        self._generate_start_state = self._generate_random_state if random_state else self._generate_fixed_state
        
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
            project=f'{self.problem_size}', 
            entity='ethanmclark1', 
            name=f'{type_name}',
            tags=[f'{problem_instance.capitalize()}'],
            )
        
        config = wandb.config
        return config
    
    # Initialize action mapping for a given problem instance
    def _init_instance(self, problem_instance: str) -> None:   
        self.instance = problems.get_instance(problem_instance)
        
    def _generate_fixed_state(self) -> tuple:
        state_proxy = np.zeros(self.state_dims, dtype=int)
        bridges = []
        num_bridges = 0
        done = False
        
        return state_proxy, bridges, num_bridges, done
    
    # Generate initial state for a given problem instance
    def _generate_random_state(self) -> tuple:
        bridge_locations = list(self.instance['mapping'].values())
        
        num_bridges = self.bridge_rng.choice(self.max_action)
        bridges = self.bridge_rng.choice(bridge_locations, size=num_bridges, replace=False).tolist()
        state = np.zeros(self.grid_dims, dtype=int)
        
        for bridge in bridges:
            state[tuple(bridge)] = 1
        
        state_proxy = self._get_state_proxy(state)
        num_bridges = np.count_nonzero(state_proxy)
        done = False
        
        return state_proxy, bridges, num_bridges, done
    
    def _get_state_proxy(self, state: np.ndarray) -> np.ndarray:
        mutable_cells = self.instance['mapping'].values()
        state_proxy = state[[cell[0] for cell in mutable_cells], [cell[1] for cell in mutable_cells]]
        return state_proxy
    
    def _get_state_from_proxy(self, state_proxy: np.ndarray) -> np.ndarray:
        state = np.zeros(self.grid_dims, dtype=int)
        mutable_cells = list(self.instance['mapping'].values())
        
        for i, cell in enumerate(state_proxy):
            state[tuple(mutable_cells[i])] = cell
        
        return state
                
    def _transform_action(self, action: int) -> tuple:
        if action == 0:
            return action

        return self.instance['mapping'][action]
    
    def _place_bridge(self, state_proxy: np.ndarray,action: int) -> np.ndarray:
        next_state_proxy = copy.deepcopy(state_proxy)
        
        if action != 0:      
            next_state_proxy[action - 1] = 1
            
        return next_state_proxy
    
    def _reassign_states(self, 
                         prev_state_proxy: np.ndarray, 
                         prev_action: int, 
                         state_proxy: np.ndarray, 
                         action: int, 
                         next_state_proxy: np.ndarray
                         ) -> tuple:
        
        action_a_success = not np.array_equal(prev_state_proxy, state_proxy)
        action_b_success = not np.array_equal(state_proxy, next_state_proxy)
        
        commutative_state_proxy = self._place_bridge(prev_state_proxy, action)
    
        if action_a_success and action_b_success:         
            pass   
        elif not action_a_success and action_b_success:
            if prev_action != action:
                next_state_proxy = commutative_state_proxy
        elif action_a_success and not action_b_success:
            commutative_state_proxy = prev_state_proxy
            next_state_proxy = state_proxy
        else:
            commutative_state_proxy = prev_state_proxy
            
        return commutative_state_proxy, next_state_proxy
    
    def _create_instance(self, state_proxy: np.ndarray) -> tuple:
        graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
        nx.set_edge_attributes(graph, 1, 'weight')
        
        state = self._get_state_from_proxy(state_proxy)
        desc = copy.deepcopy(state).reshape(self.grid_dims)
        
        return graph, desc
        
    # Cell Values: {Frozen: 0, Bridge: 1, Start: 2, Goal: 3, Hole: 4}
    def _calc_utility(self, state_proxy: np.ndarray) -> float:
        def manhattan_dist(a: tuple, b: tuple) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        utilities = []
        graph, desc = self._create_instance(state_proxy)
        
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
    
    def _get_next_state(self, state_proxy: np.ndarray, action: int) -> np.ndarray:
        next_state_proxy = copy.deepcopy(state_proxy)
        
        if action != 0 and (self.action_success_rate == 1 or self.action_success_rate >= self.action_rng.random()):
            next_state_proxy = self._place_bridge(state_proxy, action)
            
        return next_state_proxy

    def _get_reward(self, state_proxy: np.ndarray, action: int, next_state_proxy: np.ndarray, num_action: int) -> tuple:
        reward = 0
        
        done = action == 0
        timeout = num_action == self.max_action
        
        if not done:
            if not np.array_equal(state_proxy, next_state_proxy):
                util_s = self._calc_utility(state_proxy)
                util_s_prime = self._calc_utility(next_state_proxy)
                reward = util_s_prime - util_s
            reward -= self.action_cost * num_action
                        
        return reward, done or timeout
        
    def _step(self, state_proxy: np.ndarray, action: int, num_action: int) -> tuple:
        next_state_proxy = self._get_next_state(state_proxy, action)
        reward, done = self._get_reward(state_proxy, action, next_state_proxy, num_action)  
        
        return reward, next_state_proxy, done
    
    def get_adaptations(self, problem_instance: str) -> list:
        approach = self.__class__.__name__
        try:
            adaptations = self._load(problem_instance)
        except FileNotFoundError:
            print(f'No stored adaptation for {approach} on the {problem_instance.capitalize()} problem instance.')
            print('Generating new adaptation...')
            adaptations = self._generate_adaptations(problem_instance)
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