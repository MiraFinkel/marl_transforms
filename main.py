from utils import *
from visualize import *
import Environments.MultiTaxiEnv.multitaxienv.taxi_environment as taxi_env

if __name__ == '__main__':
    # TODO Guy: to add "set_display" to particle environment
    taxi_env.set_display(False)
    env_name = TAXI
    agent_name = A2C
    iteration_num = 100

    ray.init(num_gpus=NUM_GPUS, local_mode=WITH_DEBUG)
    episode_reward_mean, env, agent, config = train(env_name, agent_name, iteration_num, with_transform=False)
    taxi_env.set_display(True)
    evaluate(env, agent, config)
    ray.shutdown()

    results = [episode_reward_mean]
    names = [WITHOUT_TRANSFORM]

    plot_result_graph(results, names, "episode_reward_mean")

    # episode_reward_mean0 = train(env_name, agent_name, with_transform=True, transform_idx=0)
    # episode_reward_mean1 = train(env_name, agent_name, with_transform=True, transform_idx=1)
    # episode_reward_mean2 = train(env_name, agent_name, with_transform=True, transform_idx=2)
    # episode_reward_mean3 = train(env_name, agent_name, with_transform=True, transform_idx=3)
    # episode_reward_mean4 = train(env_name, agent_name, with_transform=True, transform_idx=4)
