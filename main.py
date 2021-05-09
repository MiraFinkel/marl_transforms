# from Environments.taxi_environment_wrapper import set_up_env_idx
import sys
import os
import shutil
import dill

from Observer.anticipated_policy_generator import *
from Transforms.taxi_transforms import *
from utils import *
from visualize import *
from Agents.RL_agents.q_learning_agents import *
import Agents.RL_agents.rl_agent as rl_agent


def run_experiment(env_name, agent_name, num_of_episodes, num_states_in_partial_policy):
    result = {}
    # get the environment
    original_env = get_env(env_name)

    # get full anticipated policy
    full_expert_policy_dict = get_expert(env_name, original_env).full_expert_policy_dict()

    # the anticipated policy (which is part of our input and defined by the user)
    anticipated_policy = sample_anticipated_policy(full_expert_policy_dict, num_states_in_partial_policy)

    # create agent
    agent = rl_agent.create_agent(original_env, agent_name)
    # train the agent in the environment
    print("Training and evaluating the agent on the original environment...")
    original_env_result_train = rl_agent.run(agent, num_of_episodes, method=TRAIN)

    # evaluate the performance of the agent
    original_env_result_evaluate = rl_agent.run(agent, num_of_episodes, method=EVALUATE)

    # check if the anticipated policy is achieved
    anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(original_env, agent, anticipated_policy)
    if anticipated_policy_achieved:
        print("The algorithm achieved the policy. We finished our work.")
        # sys.exit()

    result[ORIGINAL_ENV] = {EVALUATION_RESULTS: original_env_result_evaluate,
                            TRAINING_RESULTS: original_env_result_train,
                            SUCCESS_RATE: success_rate,
                            GOT_AN_EXPLANATION: None}

    # create a transformed environment
    transforms = set_all_possible_transforms(original_env)
    explanation = []

    transformed_env = original_env
    for params, (transform_name, transform) in transforms.items():
        # create transformed environment
        transformed_env = transform(params)

        # create agent
        agent = rl_agent.create_agent(transformed_env, agent_name)
        # train the agent in the environment
        print("\nTraining and evaluating the agent on the transformed environment -", transform_name)
        transformed_train_result = rl_agent.run(agent, num_of_episodes, method=TRAIN)
        # evaluate the performance of the agent
        transformed_evaluation_result = rl_agent.run(agent, num_of_episodes, method=EVALUATE)

        # check if the anticipated policy is achieved in trans_env
        anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(original_env, agent, anticipated_policy)

        result[transform_name] = {EVALUATION_RESULTS: transformed_evaluation_result,
                                  TRAINING_RESULTS: transformed_train_result,
                                  SUCCESS_RATE: success_rate}
        if anticipated_policy_achieved:
            explanation.append(transform_name)
            result[transform_name][GOT_AN_EXPLANATION] = True

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

def default_experiment():
    output_folder = "./output/"
    if (os.path.isdir(output_folder)):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    env_name = TAXI_EXAMPLE
    agent_name = Q_LEARNING
    num_of_episodes_per_epoch = 10000
    num_of_epochs = 5
    num_states_in_partial_policy = 5
    result = []

    for i in range(num_of_epochs):
        print('\nEpoch num:', i + 1)
        episode_result = run_experiment(env_name, agent_name, num_of_episodes_per_epoch, num_states_in_partial_policy)
        result.append(episode_result)

    # save result
    output = open(output_folder + 'all_stats' + ".pkl", 'wb')
    dill.dump(result, output)
    output.close()

    # plot
    plot_results(result, output_folder)

    return result


def different_envs_experiment():
    env_name = TAXI_EXAMPLE
    agent_name = Q_LEARNING
    num_of_episodes_per_epoch = 1000
    num_of_envs = 1
    num_states_in_partial_policy = 10
    all_env_results = {}
    all_env_test_results = {"original": [], "walls": [], "reward": [], "reward_walls": [], "fuel": [],
                            "fuel_walls": [], "fuel_reward": [], "fuel_reward_walls": []}
    all_env_evaluate_results = {"original": [], "walls": [], "reward": [], "reward_walls": [], "fuel": [],
                                "fuel_walls": [], "fuel_reward": [], "fuel_reward_walls": []}
    all_env_success_results = {"original": [], "walls": [], "reward": [], "reward_walls": [], "fuel": [],
                               "fuel_walls": [], "fuel_reward": [], "fuel_reward_walls": []}

    for i in range(num_of_envs):
        # set_up_env_idx()
        result = run_experiment(env_name, agent_name, num_of_episodes_per_epoch, num_states_in_partial_policy)
        all_env_results[i] = result
        for k, v in all_env_test_results.items():
            all_env_test_results[k].append(np.mean(result[k]['train_episode_reward_mean']))
            all_env_evaluate_results[k].append(np.mean(result[k]['evaluate_episode_reward_mean']))
            all_env_success_results[k].append(np.mean(result[k]['success_rate']))
        plot_result_graphs(agent_name, result)

        # visualize rewards and success_rate
    # names = ["original", "walls", "reward", "reward_walls", "fuel", "fuel_walls", "fuel_reward", "fuel_reward_walls"]
    # plot_graph_by_transform_name_and_env(agent_name, all_env_test_results, "300_env_test_results_5000_episodes")
    # plot_graph_by_transform_name_and_env(agent_name, all_env_evaluate_results, "300_env_evaluate_results_5000_episodes")
    # plot_success_rate_charts(names, [np.mean(v) for v in all_env_success_results.values()],
    #                          "success_rate_300_env_5000_episodes")


if __name__ == '__main__':
    default_experiment()
