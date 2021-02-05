from constants import *
from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
import ray.rllib.agents.pg as pg
import ray.rllib.agents.a3c as a3c
import ray

s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"


def get_env(env_name, with_transform=False, transform_idx=0):
    if not with_transform:
        if env_name == TAXI:
            return TaxiEnv(), TaxiEnv
    else:
        transform_env = TransformEnvironment()
        transform_env._mapping_class.set_reduction_idx(transform_idx)
        return transform_env, TransformEnvironment


def get_agent(agent_name, config, env_to_agent):
    if agent_name == PG:
        return pg.PGTrainer(config=config, env=env_to_agent)
    elif agent_name == A3C:
        return a3c.A3CTrainer(config=config, env=env_to_agent)
    elif agent_name == A2C:
        return a3c.A2CTrainer(config=config, env=env_to_agent)


def get_taxi_config(env, number_of_taxis):
    config = {}
    if number_of_taxis == 2:
        config = {'multiagent': {'policies': {'taxi_1': (None, env.obs_space, env.action_space, {'gamma': TAXI1_GAMMA}),
                                              'taxi_2': (None, env.obs_space, env.action_space, {'gamma': TAXI2_GAMMA})
                                              },
                                 "policy_mapping_fn": lambda taxi_id: taxi_id},
                  "num_gpus": NUM_GPUS,
                  "num_workers": NUM_WORKERS}

    return config


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

    TaxiEnv.set_display(True)
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
    TaxiEnv.set_display(False)
