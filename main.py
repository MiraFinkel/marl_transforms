from Environments.taxi_environment_wrapper import set_up_env_idx
from Observer.anticipated_policy_generator import sample_anticipated_policy
from Transforms.taxi_transforms import *
from utils import *
from Observer.observer_utils import *
from visualize import *
from Agents.RL_agents.q_learning_agents import *
import Agents.RL_agents.rl_agent as rl_agent
import sys


def run_experiment(env_name, agent_name, num_of_episodes, num_states_in_partial_policy):
    result = {}
    # get the environment
    env = get_env(env_name)

    # make expert
    expert = Taxi_Expert(env)
    full_expert_policy_dict = expert.full_expert_policy_dict()

    # the anticipated policy (which is part of our input and defined by the user)
    anticipated_policy = sample_anticipated_policy(full_expert_policy_dict, env, num_states_in_partial_policy)

    # anticipated_policy = {(4, 0, None, None, None, None, None, 2): 0}

    # create agent
    agent = rl_agent.create_agent(env, agent_name)
    # train the agent in the environment
    train_episode_reward_mean = rl_agent.run(agent, num_of_episodes, method=TRAIN)

    # evaluate the performance of the agent
    evaluate_episode_reward_mean = rl_agent.run(agent, num_of_episodes, method=EVALUATE)

    # check if the anticipated policy is achieved in orig_env
    anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(env, agent, anticipated_policy)
    if anticipated_policy_achieved:
        print("The algorithm achieved the policy. We finished our work.")
        # sys.exit()

    result["original"] = {"evaluate_episode_reward_mean": evaluate_episode_reward_mean,
                          "train_episode_reward_mean": train_episode_reward_mean,
                          "success_rate": success_rate,
                          "explanation": None}

    # create a transformed environment
    transforms = set_all_possible_transforms([FUELS_TRANSFORM, REWARD_TRANSFORM, NO_WALLS_TRANSFORM])
    explanation = []

    transformed_env = env
    for params, (transform_name, transform) in transforms.items():
        # create transformed environment
        transformed_env = transform(params)

        # create agent
        agent = rl_agent.create_agent(transformed_env, agent_name)
        # evaluate the performance of the agent
        transform_episode_reward_mean = rl_agent.run(agent, num_of_episodes, method=TRAIN)
        # evaluate the performance of the agent
        transform_evaluate_episode_reward_mean = rl_agent.run(agent, num_of_episodes, method=EVALUATE)

        # check if the anticipated policy is achieved in trans_env
        anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(env, agent, anticipated_policy)

        result[transform_name] = {"evaluate_episode_reward_mean": transform_evaluate_episode_reward_mean,
                                  "train_episode_reward_mean": transform_episode_reward_mean,
                                  "success_rate": success_rate}
        if anticipated_policy_achieved:
            explanation.append(transform_name)
            result[transform_name]["explanation"] = True

    if explanation is None:
        print("no explanation found - you are too dumb for our system")
    else:
        print("explanation found %s:" % explanation)

    return result


# def different_anticipated_policy_size_experiment():
#     env_name = TAXI_EXAMPLE
#     agent_name = Q_LEARNING
#     num_of_episodes = 5000
#     result = {}
#     for i in range(1, ):
#         num_states_in_partial_policy = i


def different_envs_experiment():
    # define the environment
    env_name = TAXI_EXAMPLE
    agent_name = Q_LEARNING
    num_of_episodes = 5000
    num_states_in_partial_policy = 10
    all_env_results = {}
    all_env_test_results = {"original": [], "walls": [], "reward": [], "reward_walls": [], "fuel": [],
                            "fuel_walls": [], "fuel_reward": [], "fuel_reward_walls": []}
    all_env_evaluate_results = {"original": [], "walls": [], "reward": [], "reward_walls": [], "fuel": [],
                                "fuel_walls": [], "fuel_reward": [], "fuel_reward_walls": []}
    all_env_success_results = {"original": [], "walls": [], "reward": [], "reward_walls": [], "fuel": [],
                               "fuel_walls": [], "fuel_reward": [], "fuel_reward_walls": []}

    for i in range(300):
        set_up_env_idx()
        result = run_experiment(env_name, agent_name, num_of_episodes, num_states_in_partial_policy)
        all_env_results[i] = result
        for k, v in all_env_test_results.items():
            all_env_test_results[k].append(np.mean(result[k]['train_episode_reward_mean']))
            all_env_evaluate_results[k].append(np.mean(result[k]['evaluate_episode_reward_mean']))
            all_env_success_results[k].append(np.mean(result[k]['success_rate']))

        # visualize rewards and success_rate
    names = ["original", "walls", "reward", "reward_walls", "fuel", "fuel_walls", "fuel_reward", "fuel_reward_walls"]
    plot_graph_by_transform_name_and_env(agent_name, all_env_test_results, "300_env_test_results_5000_episodes")
    plot_graph_by_transform_name_and_env(agent_name, all_env_evaluate_results, "300_env_evaluate_results_5000_episodes")
    plot_success_rate_charts(names, [np.mean(v) for v in all_env_success_results.values()],
                             "success_rate_300_env_5000_episodes")


if __name__ == '__main__':
    different_envs_experiment()
