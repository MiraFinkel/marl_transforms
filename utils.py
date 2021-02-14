from constants import *
from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
import Environments.MultiTaxiEnv.multitaxienv.taxi_environment as taxi_env


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
