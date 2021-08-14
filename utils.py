import pickle
from itertools import product

from Observer.lunar_lander_expert import LunarLanderExpert
from Observer.single_taxi_expert import SingleTaxiExpert
from Observer.taxi_expert import Taxi_Expert
from constants import *
from Transforms.transform_constants import *


def get_env(env_name, number_of_agents=1):
    """
    :param env_name:
    :param number_of_agents:
    :return:
    """
    if env_name == TAXI:
        from Environments.taxi_environment_wrapper import TaxiSimpleEnv
        return TaxiSimpleEnv()
    elif env_name == TAXI_EXAMPLE:
        from Environments.taxi_environment_wrapper import TaxiSimpleExampleEnv
        return TaxiSimpleExampleEnv()
    elif env_name == SINGLE_TAXI_EXAMPLE:
        from Environments.SingleTaxiEnv.single_taxi_wrapper import SingleTaxiSimpleEnv
        return SingleTaxiSimpleEnv()
    elif env_name == LUNAR_LANDER:
        from Environments.lunar_lander_wrapper import LunarLenderWrapper
        return LunarLenderWrapper()
    elif env_name == SPEAKER_LISTENER:
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


def get_expert(env_name, env):
    if env_name == TAXI_EXAMPLE:
        return Taxi_Expert(env)
    elif env_name == SINGLE_TAXI_EXAMPLE:
        return SingleTaxiExpert(env)
    elif env_name == LUNAR_LANDER:
        return LunarLanderExpert(env)


def is_partial_obs_equal_to_state(partial_obs, state):
    if len(partial_obs) != len(state):
        raise Exception("The length of the partial observation differs from the state")
    for i in range(len(partial_obs)):
        if partial_obs[i] is None:
            continue
        if partial_obs[i] != state[i]:
            return False
    return True


def is_actions_align(action, target_policy_actions):
    target_policy_actions = [target_policy_actions] if isinstance(target_policy_actions, int) else target_policy_actions
    for act in target_policy_actions:
        if act == action:
            return True
    return False


def print_num_of_success_failed_policies(num_of_success_policies, num_of_failed_policies):
    print("num_of_success_policies: ", num_of_success_policies)
    print("num_of_failed_policies: ", num_of_failed_policies)


def is_anticipated_policy_achieved(env, agent, anticipated_policy):
    agent.evaluating = True
    num_of_success_policies, num_of_failed_policies = 0, 0
    for partial_obs in anticipated_policy.keys():
        original_partial_obs = partial_obs
        partial_obs = list(partial_obs)
        states_from_partial_obs = env.get_states_from_partial_obs(partial_obs)
        for i, state in enumerate(states_from_partial_obs):
            action = agent.compute_action(state)
            if is_actions_align(action, anticipated_policy[original_partial_obs]):
                num_of_success_policies += 1
            else:
                num_of_failed_policies += 1

    all_policies = num_of_success_policies + num_of_failed_policies
    success_rate = num_of_success_policies / (all_policies if all_policies != 0 else 1)
    print("\nSuccess rate:", success_rate)
    agent.evaluating = False
    return success_rate > 0.8, success_rate


def get_transformed_env(env_name):
    if env_name == TAXI_EXAMPLE:
        from Transforms.taxi_transforms import TaxiTransformedEnv
        return TaxiTransformedEnv
    elif env_name == SINGLE_TAXI_EXAMPLE:
        from Transforms.single_taxi_transforms import SingleTaxiTransformedEnv
        return SingleTaxiTransformedEnv
    elif env_name == LUNAR_LANDER:
        from Transforms.lunar_lander_transforms import LunarLanderTransformedEnv
        return LunarLanderTransformedEnv


# def set_all_possible_transforms(original_env, env_name):
#     binary_permutations = ["".join(seq) for seq in product("01", repeat=original_env.transform_num)]
#     transforms = {}
#     for per in binary_permutations:
#         bool_params = tuple(True if int(dig) == 1 else False for dig in per)
#         if any(bool_params):
#             transformed_env = get_transformed_env(env_name)
#             transform_name = get_transform_name(env_name, bool_params)
#             transforms[bool_params] = (transform_name, transformed_env)
#     return transforms

def set_all_possible_transforms(original_env, env_name, anticipated_policy):
    env_preconditions = load_env_preconditions(env_name)
    basic_relevant_transforms = dict()
    for state, actions in anticipated_policy.items():
        for action in actions:
            if action not in basic_relevant_transforms:
                basic_relevant_transforms[action] = env_preconditions.not_allowed_features[action]
    preconditions_num = 0
    for action, precondition in basic_relevant_transforms.items():
        preconditions_num += sum([len(precondition[idx]) for idx in precondition.keys()])
    return basic_relevant_transforms


def load_existing_transforms(env_name, anticipated_policy):
    import os
    import re
    working_dir = "Transforms/taxi_example_data/taxi_transformed_env/"
    possible_env_files = os.listdir(working_dir)
    transform_names = []
    transformed_envs = []
    transforms = dict()
    dict_idx = -1

    for file_name in possible_env_files:
        match = re.search(r"\d", file_name)
        precondition_action = int(file_name[match.start()])
        for state, actions in anticipated_policy.items():
            for action in actions:
                if action == precondition_action:
                    dict_idx += 1
                    # precondition_idx = file_name[file_name.find("(") + 1:file_name.find(")")]
                    # precondition_val = file_name[file_name.find("[") + 1:file_name.find("]")]
                    transform_name = file_name[:-4]
                    transform_names.append(transform_name)
                    file = open(working_dir + file_name, "rb")
                    new_env = pickle.load(file)
                    transformed_envs.append(new_env)
                    transforms[dict_idx] = transform_name, new_env
    return transforms


def load_env_preconditions(env_name):
    if env_name == SINGLE_TAXI_EXAMPLE:
        a_file = open("Transforms/taxi_example_data/taxi_example_preconditions.pkl", "rb")
        preconditions = pickle.load(a_file)
        return preconditions
    else:
        raise Exception("not valid env_name")


def get_transform_name(env_name, bool_params):
    if env_name == TAXI_EXAMPLE:
        from Transforms.taxi_transforms import get_taxi_transform_name
        return get_taxi_transform_name(bool_params)
    elif env_name == SINGLE_TAXI_EXAMPLE:
        from Transforms.single_taxi_transforms import get_single_taxi_transform_name
        return get_single_taxi_transform_name(bool_params)
    elif env_name == LUNAR_LANDER:
        from Transforms.lunar_lander_transforms import get_lunar_lander_transform_name
        return get_lunar_lander_transform_name(bool_params)
