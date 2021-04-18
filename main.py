import Observer.anticipated_policy_generator as anticipated_policy_generator
from Transforms.taxi_transforms import *
from utils import *
from visualize import *
from Agents.RL_agents.q_learning_agents import *
import Agents.RL_agents.rl_agent as rl_agent
import sys

if __name__ == '__main__':
    # define the environment
    env_name = TAXI_EXAMPLE
    agent_for_policy_generator = VALUE_ITERATION
    agent_name = Q_LEARNING
    num_of_episodes = 500
    num_states_in_partial_policy = 10
    result = {}

    # get the environment
    env = get_env(env_name)

    # the anticipated policy (which is part of our input and defined by the user)
    # automatic_anticipated_policy = anticipated_policy_generator.get_automatic_anticipated_policy(env, env_name,
    #                                                                                              agent_for_policy_generator,
    #                                                                                              num_of_episodes,
    #                                                                                              num_states_in_partial_policy)

    anticipated_policy = {(4, 0, None, None, None, None, None, 2): 0}

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
        sys.exit()

    result["original"] = {"evaluate_episode_reward_mean": evaluate_episode_reward_mean,
                          "train_episode_reward_mean": train_episode_reward_mean,
                          "success_rate": success_rate}

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
        if anticipated_policy_achieved:
            explanation.append(transform_name)

        result[transform_name] = {"evaluate_episode_reward_mean": transform_evaluate_episode_reward_mean,
                                  "train_episode_reward_mean": transform_episode_reward_mean,
                                  "success_rate": success_rate}

    if explanation is None:
        print("no explanation found - you are too dumb for our system")
    else:
        print("explanation found %s:" % explanation)

    # visualize rewards and success_rate
    plot_result_graphs(agent_name, result)
