from Transforms.transform_constants import *
from experiments import *


def run_single_taxi_env():
    env_name = SINGLE_TAXI_EXAMPLE
    agent_name = Q_LEARNING
    num_states_in_partial_policy = 5
    num_of_epochs = 1
    num_of_episodes_per_epoch = 500
    default_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch, num_states_in_partial_policy)


def run_taxi_env():
    env_name = TAXI_EXAMPLE
    agent_name = KERAS_DQN
    num_states_in_partial_policy = 5
    num_of_epochs = 1
    num_of_episodes_per_epoch = 100
    different_anticipated_policy_size_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch)


def run_lunar_lander_env():
    env_name = LUNAR_LANDER
    agent_name = KERAS_DQN
    num_states_in_partial_policy = 5
    num_of_epochs = 5
    num_of_episodes_per_epoch = 500
    default_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch, num_states_in_partial_policy)


if __name__ == '__main__':
    run_single_taxi_env()
