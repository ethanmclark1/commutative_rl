# Multi-Agent Context-Dependent Language

This repository contains an implementation of [Commutative Reinforcement Learning](https://github.com/ethanmclark1/earl) for designing a language protocol that enables autonomous agents to develop grounded languages for communication. We evaluate MA-CDL on the Signal8 domain, which is a cooperative path planning task where the aerial agent assists the ground agent in navigating through a complex environment. The objective is to create a system where a fully observable aerial agent can guide a partially observable ground agent to navigate through an environment with observable and unobservable obstacles, and reach a specified goal.

The language protocol is based on partitioning the environment into segmented regions, which form the alphabet of the language. The aerial agent constructs messages using this alphabet to provide instructions to the ground agent, helping it navigate obstacles and reach the goal efficiently.

We implement Commutative Reinforcement Learning in both discretized and continuous action spaces using a Deep Q-Network (DQN) and Twin Delayed Deep Deterministic Policy Gradient (TD3), respectively.

## Visualization of Signal8 Domain

To further illustrate the [Signal8 ](https://github.com/ethanmclark1/signal8)domain, we present visualizations of the environment at inference time and the structure of two task instances:

<p align="center">
  <img src="img/Signal8.png" alt="Image 2" width="400" />
</p>

The aerial agent's perspective includes the following elements:

* Green Circle: Ground agent's starting position
* Yellow Circle: Ground agent's target position
* Big Red Circles: Large obstacles

The ground agent's perspective includes the following elements:

* Green Circle: Starting position
* Yellow Circle: Target position
* Small Red Circles: Small obstacles

**Task Instance 1: Cross**

<p align="center">
  <img src="img/cross.png" alt="Image 1" width="400" />
</p>

**Task Instance 2: Stellaris**

<p align="center">
  <img src="img/stellaris.png" alt="Image 2" width="400" />
</p>

In the illustrated task instances, the red sections represent the areas where large obstacles can be generated, while the remaining white space is available for the ground agent's starting position, target location, and small obstacles. The specific placement of these elements within the designated areas is randomized during task generation.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/ethanmclark1/ma-cdl.git
   ```
2. Navigate to the project directory:

   ```
   cd ma-cdl
   ```
3. Install the required dependencies using one of the following methods:

   a. Using `pip` and `requirements.txt`:

   ```
   pip install -r requirements.txt
   ```

   This command will install all the necessary dependencies listed in the `requirements.txt` file.

   b. Using `setup.py`:

   ```
   python setup.py install
   ```

   If you have a `setup.py` file that properly defines the project dependencies, you can use this command to install them.

   c. Using `conda` and `requirements.txt`:

   ```
   conda create --name ma-cdl python=3.11
   conda activate ma-cdl
   conda install --file requirements.txt
   ```

   If you prefer to use `conda` for managing your dependencies, you can create a new `conda` environment, activate it, and install the dependencies listed in `requirements.txt` using `conda install`.
4. Verify the installation by running the following command:

   ```
   python -c "import ma_cdl"
   ```

   If no errors are displayed, the installation was successful.

## Usage

1. Configure the desired hyperparameters and settings in the `main.py` file or through the command line.
2. Run the training script with the desired arguments:

   ```
   python main.py --num_agents 2 --num_large_obstacles 5 --num_small_obstacles 10 --seed 42 --approach commutative_dqn --problem_instance circle --random_state False --train_type online --reward_type approximate --render_mode human
   ```
3. Monitor the training progress and evaluation metrics using the specified visualization tools or logging mechanisms.
4. Retrieve the learned language for a specific problem instance:

   ```python
   from ma_cdl import MA_CDL

   ma_cdl = MA_CDL(num_agents, num_large_obstacles, num_small_obstacles, seed, random_state, train_type, reward_type, render_mode)
   language = ma_cdl.retrieve_language(approach, problem_instance)
   ```
5. Evaluate the constructed language on a specific problem instance using the `ma_cdl.evaluate()` function:

   ```python
   language_set = {
       'rl': language,
       'voronoi_map': voronoi_map_language,
       'grid_world': grid_world_language,
       'direct_path': direct_path_language
   }
   num_episodes = 100
   language_safety, ground_agent_success, avg_direction_len = ma_cdl.evaluate(problem_instance, language_set, num_episodes)
   ```

   The `evaluate()` function takes the problem instance, a dictionary of languages (including the learned language and baseline languages), and the number of episodes to evaluate. It returns the language safety, ground agent success rate, and average direction length for each approach.

The `main.py` file contains the argument definitions and their default values. You can modify this file or specify the desired hyperparameters directly through the command line when running the script. Use the appropriate flags followed by their values to customize the settings. For example:

- `--num_agents`: Set the number of ground agents in the environment.
- `--num_large_obstacles`: Set the number of large obstacles (observable only to aerial agent) in the environment.
- `--num_small_obstacles`: Set the number of small obstacles (observable only to ground agent) in the environment.
- `--seed`: Set the random seed for reproducibility.
- `--approach`: Choose the approach for language development (e.g., "commutative_dqn" or "basic_td3").
- `--problem_instance`: Specify the problem instance to train or evaluate on (e.g., "circle" or "staggered").
- `--random_state`: Determine whether to use random initial states (True or False).
- `--train_type`: Choose the training type (e.g., "online" or "offline").
- `--reward_type`: Specify the reward type to learn from (e.g., "true" or "approximate").
- `--render_mode`: Set the rendering mode for visualization (e.g., "human" or "rgb_array").

Feel free to adjust the available hyperparameters and their flags based on your specific implementation and requirements.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to acknowledge the contributions of the following individuals and resources:

- [List any notable contributors or sources of inspiration]
