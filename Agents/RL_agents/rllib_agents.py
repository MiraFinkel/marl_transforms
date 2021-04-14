RLLIB_A2C = "rllib_a2c"
RLLIB_A3C = "rllib_a3c"
RLLIB_BC = "rllib_bc"
RLLIB_DQN = "rllib_dqn"
RLLIB_APEX_DQN = "rllib_apex_dqn"
RLLIB_IMPALA = "rllib_impala"
RLLIB_MARWIL = "rllib_marwil"
RLLIB_PG = "rllib_pg"
RLLIB_PPO = "rllib_ppo"
RLLIB_APPO = "rllib_appo"
RLLIB_SAC = "rllib_sac"
RLLIB_LIN_UCB = "rllib_lin_usb"
RLLIB_LIN_TS = "rllib_lin_ts"

NUM_WORKERS = 1
WITH_DEBUG = True
NUM_GPUS = 0
g_config = {}


TAXI = "taxi"
SPEAKER_LISTENER = "simple_speaker_listener"

agents_gamma = {'taxi_1': 0.85, 'taxi_2': 0.95, 'taxi_3': 0.85, 'taxi_4': 0.95}


def is_rllib_agent(agent_name):
    return agent_name == RLLIB_A2C or \
           agent_name == RLLIB_A3C or \
           agent_name == RLLIB_BC or \
           agent_name == RLLIB_DQN or \
           agent_name == RLLIB_APEX_DQN or \
           agent_name == RLLIB_IMPALA or \
           agent_name == RLLIB_MARWIL or \
           agent_name == RLLIB_PG or \
           agent_name == RLLIB_PPO or \
           agent_name == RLLIB_APPO or \
           agent_name == RLLIB_SAC or \
           agent_name == RLLIB_LIN_UCB or \
           agent_name == RLLIB_LIN_TS


def get_config(env_name, env, number_of_agents):
    """
    :param env_name
    :param env:
    :param number_of_agents:
    :return:
    """
    config = {}
    if env_name == TAXI:
        if number_of_agents == 1:  # single agent config
            config = {"num_gpus": NUM_GPUS, "num_workers": NUM_WORKERS}
        else:  # multi-agent config
            policies = get_multi_agent_policies(env, number_of_agents)
            config = {'multiagent': {'policies': policies, "policy_mapping_fn": lambda taxi_id: taxi_id},
                      "num_gpus": NUM_GPUS,
                      "num_workers": NUM_WORKERS}
    elif env_name == SPEAKER_LISTENER:
        config = {
            "num_gpus": 0,
            "lr_schedule": [[0, 0.007], [20000000, 0.0000000001]],
            "framework": "torch",
            "env_config": {"name": "simple_speaker_listener"},
            "clip_rewards": True,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 20,
            "monitor": True,
        }
    global g_config
    g_config = config
    return config


def get_multi_agent_policies(env, number_of_agents):
    policies = {}
    for i in range(number_of_agents):
        name = 'taxi_' + str(i + 1)
        policies[name] = (None, env.observation_space, env.action_space, {'gamma': agents_gamma[name]})
    return policies
