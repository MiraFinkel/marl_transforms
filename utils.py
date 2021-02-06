import ray
from constants import *
from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
import Environments.MultiTaxiEnv.multitaxienv.taxi_environment as taxi_env

s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"
env = None
agent = None
config = None


def get_env(env_name, with_transform=False, transform_idx=0):
    """
    TODO Guy: to expand the function to work with particle environment
    :param env_name:
    :param with_transform:
    :param transform_idx:
    :return:
    """
    global env
    if not with_transform:
        if env_name == TAXI:
            env = TaxiEnv()
            return env, TaxiEnv
    else:
        transform_env = TransformEnvironment()
        env = TransformEnvironment()  # TODO Mira - do I need the transformed environment? or the original one?
        transform_env._mapping_class.set_reduction_idx(transform_idx)
        return transform_env, TransformEnvironment


def get_agent(agent_name, config, env_to_agent):
    global agent
    if agent_name == A2C:
        import ray.rllib.agents.a3c as a2c
        agent = a2c.A2CTrainer(config=config, env=env_to_agent)
    elif agent_name == A3C:
        import ray.rllib.agents.a3c as a3c
        agent = a3c.A3CTrainer(config=config, env=env_to_agent)
    elif agent_name == BC:
        import ray.rllib.agents.marwil as bc
        agent = bc.BCTrainer(config=config, env=env_to_agent)
    elif agent_name == DQN:
        import ray.rllib.agents.dqn as dqn
        agent = dqn.DQNTrainer(config=config, env=env_to_agent)
    elif agent_name == APEX_DQN:
        import ray.rllib.agents.dqn as dqn
        agent = dqn.ApexTrainer(config=config, env=env_to_agent)
    elif agent_name == IMPALA:
        import ray.rllib.agents.impala as impala
        agent = impala.ImpalaTrainer(config=config, env=env_to_agent)
    elif agent_name == MARWIL:
        import ray.rllib.agents.marwil as marwil
        agent = marwil.MARWILTrainer(config=config, env=env_to_agent)
    elif agent_name == PG:
        import ray.rllib.agents.pg as pg
        agent = pg.PGTrainer(config=config, env=env_to_agent)
    elif agent_name == PPO:
        import ray.rllib.agents.ppo as ppo
        agent = ppo.PPOTrainer(config=config, env=env_to_agent)
    elif agent_name == APPO:
        import ray.rllib.agents.ppo as ppo
        agent = ppo.APPOTrainer(config=config, env=env_to_agent)
    elif agent_name == SAC:
        import ray.rllib.agents.sac as sac
        agent = sac.SACTrainer(config=config, env=env_to_agent)
    elif agent_name == LIN_UCB:
        import ray.rllib.contrib.bandits.agents.lin_ucb as lin_ucb
        agent = lin_ucb.LinUCBTrainer(config=config, env=env_to_agent)
    elif agent_name == LIN_TS:
        import ray.rllib.contrib.bandits.agents.lin_ts as lin_ts
        agent = lin_ts.LinTSTrainer(config=config, env=env_to_agent)
    else:
        raise Exception("Not valid agent name")
    return agent


def get_config(env, number_of_taxis):
    """
    TODO Guy: to expand the function to work with particle environment
    :param env:
    :param number_of_taxis:
    :return:
    """
    global config
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


def evaluate():
    print()
    print(" ===================================================== ")
    print(" ================ STARTING EVALUATION ================ ")
    print(" ===================================================== ")
    print()

    global env, agent, config
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


def get_initialized_env():
    global env
    return env


def set_initialized_env(new_env):
    global env
    env = new_env


def get_initialized_agent():
    global agent
    return agent


def set_initialized_agent(new_agent):
    global agent
    agent = new_agent


def get_initialized_config():
    global config
    return config


def set_initialized_config(new_config):
    global config
    config = new_config
