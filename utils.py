import ray
from constants import *
from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
import Environments.MultiTaxiEnv.multitaxienv.taxi_environment as taxi_env

s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"


def get_env(env_name, with_transform=False, transform_idx=0):
    """
    TODO Guy: to expand the function to work with particle environment
    :param env_name:
    :param with_transform:
    :param transform_idx:
    :return:
    """
    if not with_transform:
        if env_name == TAXI:
            return TaxiEnv(), TaxiEnv
    else:
        transform_env = TransformEnvironment()
        transform_env._mapping_class.set_reduction_idx(transform_idx)
        return transform_env, TransformEnvironment


def get_agent(agent_name, config, env_to_agent):
    if agent_name == A2C:
        import ray.rllib.agents.a3c as a2c
        return a2c.A2CTrainer(config=config, env=env_to_agent)
    elif agent_name == A3C:
        import ray.rllib.agents.a3c as a3c
        return a3c.A3CTrainer(config=config, env=env_to_agent)
    elif agent_name == BC:
        import ray.rllib.agents.marwil as bc
        return bc.BCTrainer(config=config, env=env_to_agent)
    elif agent_name == DQN:
        import ray.rllib.agents.dqn as dqn
        return dqn.DQNTrainer(config=config, env=env_to_agent)
    elif agent_name == APEX_DQN:
        import ray.rllib.agents.dqn as dqn
        return dqn.ApexTrainer(config=config, env=env_to_agent)
    elif agent_name == IMPALA:
        import ray.rllib.agents.impala as impala
        return impala.ImpalaTrainer(config=config, env=env_to_agent)
    elif agent_name == MARWIL:
        import ray.rllib.agents.marwil as marwil
        return marwil.MARWILTrainer(config=config, env=env_to_agent)
    elif agent_name == PG:
        import ray.rllib.agents.pg as pg
        return pg.PGTrainer(config=config, env=env_to_agent)
    elif agent_name == PPO:
        import ray.rllib.agents.ppo as ppo
        return ppo.PPOTrainer(config=config, env=env_to_agent)
    elif agent_name == APPO:
        import ray.rllib.agents.ppo as ppo
        return ppo.APPOTrainer(config=config, env=env_to_agent)
    elif agent_name == SAC:
        import ray.rllib.agents.sac as sac
        return sac.SACTrainer(config=config, env=env_to_agent)
    elif agent_name == LIN_UCB:
        import ray.rllib.contrib.bandits.agents.lin_ucb as lin_ucb
        return lin_ucb.LinUCBTrainer(config=config, env=env_to_agent)
    elif agent_name == LIN_TS:
        import ray.rllib.contrib.bandits.agents.lin_ts as lin_ts
        return lin_ts.LinTSTrainer(config=config, env=env_to_agent)
    else:
        raise Exception("Not valid agent name")


def get_config(env, number_of_taxis):
    """
    TODO Guy: to expand the function to work with particle environment
    :param env:
    :param number_of_taxis:
    :return:
    """
    config = {}
    if number_of_taxis == 2:
        config = {'multiagent': {'policies': {'taxi_1': (None, env.obs_space, env.action_space, {'gamma': TAXI1_GAMMA}),
                                              'taxi_2': (None, env.obs_space, env.action_space, {'gamma': TAXI2_GAMMA})
                                              },
                                 "policy_mapping_fn": lambda taxi_id: taxi_id},
                  "num_gpus": NUM_GPUS,
                  "num_workers": NUM_WORKERS}

    return config


def train(env_name, agent_name, iteration_num, with_transform=False, transform_idx=0):
    env, env_to_agent = get_env(env_name, with_transform, transform_idx)
    config = get_config(env, NUM_TAXI)
    agent = get_agent(agent_name, config, env_to_agent)
    episode_reward_mean = []
    for it in range(iteration_num):
        result = agent.train()
        print(s.format(
            it + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"]
        ))
        episode_reward_mean.append(result["episode_reward_mean"])

    return episode_reward_mean, env, agent, config


def evaluate(env, agent, config):
    print()
    print(" ===================================================== ")
    print(" ================ STARTING EVALUATION ================ ")
    print(" ===================================================== ")
    print()

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
