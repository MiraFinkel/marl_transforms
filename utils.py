from constants import *
from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
import Environments.MultiTaxiEnv.multitaxienv.taxi_environment as taxi_env

# s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"

env = None
agent = None
config = None


def get_env(env_name, number_of_agents=1, with_transform=False, transform_idxes=None):
    """
    TODO Guy: to expand the function to work with particle environment
    :param env_name:
    :param with_transform:
    :param transform_idxes:
    :return:
    """
    global env
    if not with_transform:
        if env_name == TAXI:
            taxi_env.set_number_of_agents(number_of_agents)
            env = TaxiEnv()
            return env, TaxiEnv
    else:
        taxi_env.set_number_of_agents(number_of_agents)
        transform_env = TransformEnvironment()
        env = TransformEnvironment()  # TODO Mira - do I need the transformed environment? or the original one?
        transform_env._mapping_class.set_reduction_idx(transform_idxes)
        return transform_env, TransformEnvironment


def get_multi_agent_policies(env, number_of_agents):
    policies = {}
    for i in range(number_of_agents):
        name = 'taxi_' + str(i + 1)
        policies[name] = (None, env.observation_space, env.action_space, {'gamma': agents_gamma[name]})
    return policies


def get_config(env_name, env, number_of_agents):
    """
    TODO Guy: to expand the function to work with particle environment
    :param env:
    :param number_of_agents:
    :return:
    """
    global config
    config = {}
    if env_name == TAXI:
        if number_of_agents == 1:  # single agent config
            config = {"num_gpus": NUM_GPUS, "num_workers": NUM_WORKERS}
        else:  # multi-agent config
            policies = get_multi_agent_policies(env, number_of_agents)
            config = {'multiagent': {'policies': policies, "policy_mapping_fn": lambda taxi_id: taxi_id},
                      "num_gpus": NUM_GPUS,
                      "num_workers": NUM_WORKERS}
    return config


