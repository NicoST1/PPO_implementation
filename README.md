# Proximal Policy Optimization (PPO) Implementation

This repository hosts a simple and intuitive implementation of Proximal Policy Optimization (PPO), a class of reinforcement learning algorithms. The implementation is compatible with the OpenAI Gym library environments. For testing purposes, we're utilizing the Slime Volleyball environment.

## Installation

Before using this repository, you need to install the required packages mentioned in the `requirements.txt` file.
```
pip install -r requirements.txt
```
Python 3.10 was used.

## Usage

The Jupyter notebook `main.ipynb` contains four main sections:

1. **Tune Hyperparameters**: This section includes the code needed to tune parameters such as learning rate, gamma, etc., using the `hyperopt` library.

2. **Train PPO**: This section provides the code to train the PPO actor and critic networks.

3. **Test Current PPO**: If you have trained your own PPO, you can use this section to test your model.

4. **Test Pretrained PPO**: If you wish to test the pretrained PPO model, you can use this section.


