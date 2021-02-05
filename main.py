import ray
from utils import *
from visualize import *
import Environments.MultiTaxiEnv.multitaxienv.taxi_environment as taxi_env

s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"


def train(env_name, agent_name, with_transform=False, transform_idx=0):
    env, env_to_agent = get_env(env_name, with_transform, transform_idx)
    ray.init(num_gpus=NUM_GPUS, local_mode=WITH_DEBUG)
    config = get_taxi_config(env, NUM_TAXI)
    agent = get_agent(agent_name, config, env_to_agent)
    episode_reward_mean = []
    for it in range(ITER_NUM):
        result = agent.train()
        print(s.format(
            it + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"]
        ))
        episode_reward_mean.append(result["episode_reward_mean"])
    evaluate(env, agent, config)
    ray.shutdown()
    return episode_reward_mean


def evaluate(env, agent, config):
    print(" ===================================================== ")
    print(" ================ STARTING EVALUATION ================ ")
    print(" ===================================================== ")

    taxi_env.set_display(True)
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = {}
        for agent_id, agent_obs in obs.items():
            policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
            action[agent_id] = agent.compute_action(agent_obs, policy_id=policy_id)
        obs, reward, done, info = env.step(action)
        done = done['__all__']
        # sum up reward for all agents
        episode_reward += sum(reward.values())
    print(episode_reward)
    taxi_env.set_display(False)


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
