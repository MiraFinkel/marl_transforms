from Observer.lunar_lander_expert import LunarLanderExpert
from Observer.taxi_expert import Taxi_Expert
from constants import *


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
