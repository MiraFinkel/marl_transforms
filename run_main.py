from experiments import *

if __name__ == '__main__':
    env_name = LUNAR_LANDER
    agent_name = Q_LEARNING
    num_states_in_partial_policy = 5
    num_of_epochs = 1
    num_of_episodes_per_epoch = 500
    different_anticipated_policy_size_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch)
