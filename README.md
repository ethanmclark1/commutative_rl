# Commutative Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python 3.7+](https://img.shields.io/badge/python-3.10+-blue.svg) ![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the implementation of **Commutative Reinforcement Learning** (Commutative RL), a novel approach designed to exploit order-invariance in reinforcement learning problems. This code accompanies the thesis "Reinforcement Learning for Commutative Markov Decision Processes."

## Introduction

Traditional reinforcement learning (RL) algorithms implicitly assume that the order of actions matters, treating each permutation of the same action set as a unique trajectory. However, in many real-world problems, such as resource allocation, portfolio optimization, and infrastructure planning, the final outcome depends only on which actions are taken, not the sequence in which they are performed.

Commutative RL exploits this order-invariance property to achieve significant computational efficiency improvements, reducing complexity from exponential $O(|A|^h)$ to polynomial $O(h^{|A|-1})$ in the problem horizon where $|A|$ represents the number of actions and $h$ represents the horizon length.

### Key Concepts

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
cd commutative_rl
```

2. Install dependencies:

```bash
pip install numpy pyyaml wandb
```

## Usage

### Basic Usage

To run experiments with default settings:

```bash
python commutative_rl/main.py
```

This will run the default QTable approach on the first problem instance.

### Advanced Usage

Specify approaches and problem instances:

```bash
python commutative_rl/main.py --approaches QTable SuperActionQTable CombinedRewardQTable HashMapQTable --problem_instances instance_0 instance_1
```

Customize hyperparameters:

```bash
python commutative_rl/main.py --approaches SuperActionQTable --alpha 0.1 --epsilon 0.2 --gamma 0.95 --max_noise 5
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

## Component Selection Problem

The Component Selection task presents a fundamental order-invariant optimization problem where an agent must reach a target sum by selecting values from a fixed set of numbers. While simplified, this task captures core characteristics found in resource allocation and portfolio optimization problems.

### Environment Details

* Fixed set of continuous-valued elements randomly selected from a range
* Each element has an associated cost
* Elements can be selected multiple times (with random noise added to the selected value)
* Reward structure incentivizes precise target matching while penalizing overshooting
* Episode terminates when the agent selects the null action or the cumulative sum exceeds a threshold

## Results

In the Component Selection domain, Commutative RL implementations demonstrate significant advantages:

* **Combined Reward** : Up to 16.7× faster convergence compared to standard Q-learning
* **Hash Map** : 4.2× faster convergence with 10% higher return
* **Super Action** : 1.06× faster convergence with 10% higher return

These results validate the theoretical foundations of Commutative MDPs and demonstrate the practical advantages of exploiting action commutativity in order-invariant domains.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
