from utils import *
from visualize import *




if __name__ == '__main__':
    taxi_env.set_display(False)
    env_name = TAXI
    agent_name = AGENT
    iteration_num = ITER_NUM

    episode_reward_mean = train(env_name, agent_name, with_transform=False)
    # episode_reward_mean0 = train(env_name, agent_name, with_transform=True, transform_idx=0)
    # episode_reward_mean1 = train(env_name, agent_name, with_transform=True, transform_idx=1)
    # episode_reward_mean2 = train(env_name, agent_name, with_transform=True, transform_idx=2)
    # episode_reward_mean3 = train(env_name, agent_name, with_transform=True, transform_idx=3)
    # episode_reward_mean4 = train(env_name, agent_name, with_transform=True, transform_idx=4)

    results = [episode_reward_mean]
    names = [WITHOUT_TRANSFORM]

    plot_result_graph(results, names, "episode_reward_mean")
