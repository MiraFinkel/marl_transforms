from Transforms.taxi_transforms import TaxiInfiniteFuelTransform
from constants import *
import numpy as np


def get_env(env_name, number_of_agents=1):
    """
    TODO Guy: to expand the function to work with particle environment
    :param env_name:
    :param number_of_agents:
    :return:
    """
    if env_name == TAXI:
        from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
        import Environments.MultiTaxiEnv.multitaxienv.taxi_environment as taxi_env
        taxi_env.set_number_of_agents(number_of_agents)
        env = TaxiEnv()
        return env, TaxiEnv
    if env_name == SPEAKER_LISTENER:
        from supersuit import pad_observations_v0, pad_action_space_v0
        from pettingzoo.mpe import simple_speaker_listener_v3
        from ray.tune.registry import register_env
        from ray.rllib.env import PettingZooEnv

        def create_env(args):
            env = simple_speaker_listener_v3.env()
            env = pad_action_space_v0(env)
            env = pad_observations_v0(env)
            return env

        get_env_lambda = lambda config: PettingZooEnv(create_env(config))
        register_env(SPEAKER_LISTENER, lambda config: get_env_lambda(config))
        return get_env_lambda({}), SPEAKER_LISTENER


def is_partial_obs_equal_to_state(partial_obs, state):
    if len(partial_obs) != len(state):
        raise Exception("The length of the partial observation differs from the state")
    for i in range(len(partial_obs)):
        if partial_obs[i] is None:
            continue
        if partial_obs[i] != state[i]:
            return False
    return True


def target_policy_achieved(env, agent, target_policy):  # TODO Mira - to add the multi agent case
    num_of_success_policies = 0
    num_of_failed_policies = 0
    result = True
    for partial_obs in target_policy.keys():
        original_partial_obs = partial_obs
        partial_obs = list(partial_obs)
        states_from_partial_obs = env.get_states_from_partial_obs(partial_obs)
        for state in states_from_partial_obs:
            state = np.reshape(np.array(state), (1, len(state)))
            action = agent.compute_action(state)
            if action != target_policy[original_partial_obs]:
                num_of_failed_policies += 1
                result = False
            else:
                num_of_success_policies += 1
    print(" ============================================================== ")
    print("=========> num_of_success_policies: ", num_of_success_policies)
    print("=========> num_of_failed_policies: ", num_of_failed_policies)
    return result
