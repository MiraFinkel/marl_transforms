from constants import *
from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
import Environments.MultiTaxiEnv.multitaxienv.taxi_environment as taxi_env
from supersuit import pad_observations_v0, pad_action_space_v0
from pettingzoo.mpe import simple_speaker_listener_v3
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
import numpy as np

TAXI = "taxi"
SPEAKER_LISTENER = "simple_speaker_listener"

def get_env(env_name, number_of_agents=1, with_transform=False, transform=None, transform_idxes=None):
    """
    TODO Guy: to expand the function to work with particle environment
    :param env_name:
    :param number_of_agents:
    :param with_transform:
    :param transform:
    :param transform_idxes:
    :return:
    """
    global env
    if env_name == TAXI:
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
    if env_name == SPEAKER_LISTENER:
        from supersuit import pad_observations_v0, pad_action_space_v0
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
    print(" ============================================================== ")
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
                # print("[FAIL]: action: {}, target_policy[partial_obs]: {}".format(action, target_policy[original_partial_obs]))
                # print("[FAIL]: state: {}".format(state))
                num_of_failed_policies += 1
                result = False
            else:
                # print("[SUCCESS]: action: {}, target_policy[partial_obs]: {}".format(action, target_policy[original_partial_obs]))
                # print("[SUCCESS]: state: {}".format(state))
                num_of_success_policies += 1
    print(" ============================================================== ")
    print("=========> num_of_success_policies: ", num_of_success_policies)
    print("=========> num_of_failed_policies: ", num_of_failed_policies)
    return result
