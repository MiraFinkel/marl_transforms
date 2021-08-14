# from Environments.taxi_environment_wrapper import set_up_env_idx
import sys
import os
import shutil
import dill
import pickle
from Observer.anticipated_policy_generator import *
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
    anticipated_policy = dict()
    anticipated_policy[(2, 0, 0, 3, None)] = [1]
    anticipated_policy[(1, 0, 0, 3, None)] = [1]
    anticipated_policy[(0, 0, 0, 3, None)] = [4]

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
    # transforms = set_all_possible_transforms(original_env, env_name, anticipated_policy)
    transforms = load_existing_transforms(env_name, anticipated_policy)
    explanation = []

    transformed_env = original_env
    for params, (transform_name, transformed_env) in transforms.items():
        # create transformed environment
        # transformed_env = transform(params)

        # create agent
        agent = rl_agent.create_agent(transformed_env, agent_name)
        # train the agent in the environment
        print("\nTraining and evaluating the agent on the transformed environment -", transform_name)
        transformed_train_result = rl_agent.run(agent, num_of_episodes, method=TRAIN)
        # evaluate the performance of the agent
        transformed_evaluation_result = rl_agent.run(agent, num_of_episodes, method=EVALUATE)

        # check if the anticipated policy is achieved in trans_env
        anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(original_env, agent,
                                                                                   anticipated_policy)

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


def different_anticipated_policy_size_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch):
    # num_states = [1, 5, 10, 20, 30]
    num_states = [5]
    for num_states_in_partial_policy in num_states:
        default_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch, num_states_in_partial_policy)


def plot_results_by_number_of_transforms(pkl_name, output_folder, num_episodes):
    with open(pkl_name, 'rb') as f:
        data = pickle.load(f)
        result_1, result_2, result_3, result_4, result_5 = {}, {}, {}, {}, {}
        ls1, ls2, ls3, ls4, ls5 = [], [], [], [], []
        for cur_dict in data:
            for k, v in cur_dict.items():
                name = k.split('_')[:-1]
                if k == ORIGINAL_ENV:
                    orig_res = prepare_calc_mean(v)
                    result_1[k] = orig_res
                    result_2[k] = orig_res
                    result_3[k] = orig_res
                    result_4[k] = orig_res
                    result_5[k] = orig_res
                elif len(name) == 1:
                    result_1[k] = prepare_calc_mean(v)
                elif len(name) == 2:
                    result_2[k] = prepare_calc_mean(v)
                elif len(name) == 3:
                    result_3[k] = prepare_calc_mean(v)
                elif len(name) == 4:
                    result_4[k] = prepare_calc_mean(v)
                elif len(name) == 5:
                    result_5[k] = prepare_calc_mean(v)
            ls1.append(result_1)
            ls2.append(result_2)
            ls3.append(result_3)
            ls4.append(result_4)
            ls5.append(result_5)
            result_1, result_2, result_3, result_4, result_5 = {}, {}, {}, {}, {}
        save_cur_fig = False
        plot_results(ls1, output_folder, file_name="_1trans_" + str(num_episodes) + "_", save_fig=save_cur_fig)
        plot_results(ls2, output_folder, file_name="_2trans_" + str(num_episodes) + "_", save_fig=save_cur_fig)
        plot_results(ls3, output_folder, file_name="_3trans_" + str(num_episodes) + "_", save_fig=save_cur_fig)
        plot_results(ls4, output_folder, file_name="_4trans_" + str(num_episodes) + "_", save_fig=save_cur_fig)
        plot_results(ls5, output_folder, file_name="_5trans_" + str(num_episodes) + "_", save_fig=save_cur_fig)


def default_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch, num_states_in_partial_policy):
    output_folder = "./output/"
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    result = []

    for i in range(num_of_epochs):
        print('\nEpoch num:', i + 1)
        episode_result = run_experiment(env_name, agent_name, num_of_episodes_per_epoch, num_states_in_partial_policy)
        result.append(episode_result)

    # save result
    pkl_name = output_folder + agent_name + '_all_stats_' + str(num_of_episodes_per_epoch) + "_states_" + str(
        num_states_in_partial_policy) + ".pkl"
    output = open(pkl_name, 'wb')
    dill.dump(result, output)
    output.close()

    # plot
    # plot_results(result, output_folder)
    # plot_results_by_number_of_transforms(pkl_name, output_folder, num_of_episodes_per_epoch)
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


