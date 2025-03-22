# Commutative Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python 3.7+](https://img.shields.io/badge/python-3.10+-blue.svg) ![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the implementation of **Commutative Reinforcement Learning** (Commutative RL) applied to an Environment Reconfiguration problem in the form of city planning. This code accompanies the thesis "Reinforcement Learning for Commutative Markov Decision Processes."

## Introduction

Traditional reinforcement learning (RL) algorithms implicitly assume that the order of actions matters, treating each permutation of the same action set as a unique trajectory. However, in many real-world problems like city planning and infrastructure development, the final outcome depends only on which actions are taken, not the sequence in which they are performed.

Commutative RL exploits this order-invariance property to achieve significant computational efficiency improvements, reducing complexity from exponential $O(|A|^h)$ to polynomial $O(h^{|A|-1})$ in the problem horizon where $|A|$ represents the number of actions and $h$ represents the horizon length.

### Key Concepts

* **Environment Reconfiguration (ER)** : A meta-optimization problem where the environment itself becomes the optimization objective through strategic modifications.
* **Commutative Markov Decision Processes (Commutative MDPs)** : A class of reinforcement learning problems where final states and rewards depend only on which actions are taken, not their order.
* **Order-Invariance** : The property that action permutations yield identical outcomes.
* **Sample Efficiency** : Commutative RL approaches can converge significantly faster than traditional RL methods in suitable domains.

## Repository Structure

```
commutative_rl/
├── agents/                # Agent implementations
│   ├── utils/             # Helper utilities
│   ├── commutative.py     # Commutative RL implementations
│   └── traditional.py     # Traditional RL implementations
├── problems/              # Problem definitions
│   ├── problem_generator.py  # Generates random problem instances
│   └── problems.yaml      # Pre-defined problem instances
├── tests/                 # Test suite
├── env.py                 # Environment implementation
├── arguments.py           # Command-line argument parser
└── main.py                # Main script to run experiments
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ethanmclark1/commutative_rl.git
git checkout frozen_lake
cd commutative_rl
```

2. Install dependencies using the requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To run experiments with default settings:

```bash
python commutative_rl/main.py
```

This will run the default QTable approach on all problem instances.

### Advanced Usage

Specify approaches and problem instances:

```bash
python commutative_rl/main.py --approaches QTable SuperActionQTable CombinedRewardQTable HashMapQTable --problem_instances instance_0 instance_1
```

Customize environment parameters:

```bash
python commutative_rl/main.py --approaches SuperActionQTable --grid_dims 16x16 --n_bridges 16 --n_starts 4 --n_goals 4
```

Modify agent hyperparameters:

```bash
python commutative_rl/main.py --approaches HashMapQTable --alpha 0.02 --epsilon 0.3 --gamma 0.98
```

Generate a specific number of problem instances:

```bash
python commutative_rl/main.py --n_instances 10 --approaches HashMapQTable
```

### Commutative RL Implementations

The repository provides three distinct Commutative RL implementations:

1. **Super Action** (`SuperActionQTable`): Treats consecutive actions as pairs and updates their combined Q-value.
2. **Combined Reward** (`CombinedRewardQTable`): Aggregates rewards from permutation-equivalent action sequences.
3. **Hash Map** (`HashMapQTable`): Uses a mapping function that grounds commutative updates in the environment dynamics.

Each implementation enforces the structural constraint that guarantees $Q(s, ⟨a,b⟩) = Q(s, ⟨b,a⟩)$ for any action pair.

### Baseline Implementations

For comparison, the repository also includes:

1. **Q-Learning** (`QTable`): Traditional Q-learning.
2. **Triple Data Q-Learning** (`TripleDataQTable`): Traditional Q-learning with 3× training samples.

## City Planning Environment

The City Planning environment extends the Frozen Lake domain to incorporate stochastic action outcomes and dynamic travel patterns, creating a realistic urban development optimization problem. This environment represents an Environment Reconfiguration (ER) problem, where the agent modifies the environment's structure to optimize path efficiency.

### Environment Details

* 12×12 grid with fixed start and goal locations representing residential and commercial zones
* Strategic bridge placement over "holes" (water/traffic bottlenecks)
* Stochastic bridge construction (65% success rate per attempt)
* Multi-stage bridge building (requires 2 successful attempts to complete)
* Path optimization considering all possible source-target pairs
* Action costs (bridge construction costs) randomly assigned from [0,1]
* Rewards based on improvement in average shortest path length minus construction costs
* Terminal action allows the agent to "finish" with a reward based on final path efficiency

The environment naturally exhibits action commutativity since the final state and cumulative reward depend only on which bridges are successfully built, not the order in which construction was attempted.

## Results

In the City Planning domain, Commutative RL implementations demonstrate the following performance:

* **Hash Map** : 1.17× faster convergence with 3.3% higher return than standard Q-learning
* **Combined Reward** : 1.04× faster convergence with 2.6% higher return
* **Super Action** : Failed to converge within experimental timeframe

The more modest improvements compared to the Component Selection domain are explained by the theoretical relationship between horizon length and action space size. In City Planning, the action space ($|A|=13$) exceeds the typical horizon length ($h≈7$), which is less favorable for Commutative RL's complexity reduction.

## License

This project is licensed under the MIT License - see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@thesis{clark2025commutative,
  title={Reinforcement Learning for Commutative Markov Decision Processes},
  author={Ethan M. Clark},
  year={2025},
  school={Arizona State University}
}
```
