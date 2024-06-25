# Hyperparameter Optimization for Neural Networks

This repository contains various methods for hyperparameter optimization of neural networks, including Grid Search, Random Search, Bayesian Optimization, Genetic Algorithm, Particle Swarm Optimization, and Differential Evolution.

## Installation

First, install the required libraries by running the following command:

```sh
pip install -r requirements.txt
```

## Usage
1. Run the DataSelect.py script to preprocess the data:

```sh
python DataSelect.py
```

2. After preprocessing the data, you can run any of the optimization scripts:

GridSearch.py for Grid Search
RandomSearch.py for Random Search
BayesianOptimization.py for Bayesian Optimization
GeneticAlgorithm.py for Genetic Algorithm
ParticleSwarmOptimization.py for Particle Swarm Optimization
DifferentialEvolution.py for Differential Evolution

Example:

```sh
python GridSearch.py
```

Replace GridSearch.py with the appropriate script name as needed.

## Description of Files

DataSelect.py: Script for data preprocessing.
GridSearch.py: Script for hyperparameter optimization using Grid Search.
RandomSearch.py: Script for hyperparameter optimization using Random Search.
BayesianOptimization.py: Script for hyperparameter optimization using Bayesian Optimization.
GeneticAlgorithm.py: Script for hyperparameter optimization using Genetic Algorithm.
ParticleSwarmOptimization.py: Script for hyperparameter optimization using Particle Swarm Optimization.
DifferentialEvolution.py: Script for hyperparameter optimization using Differential Evolution.
