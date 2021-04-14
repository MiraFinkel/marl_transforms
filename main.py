import Observer.anticipated_policy_generator as anticipated_policy_generator
from Transforms.taxi_transforms import *
from utils import *
from visualize import *
from Agents.RL_agents.rl_agent import *
from Agents.RL_agents.q_learning_agents import *
import Agents.RL_agents.rl_agent as rl_agent

if __name__ == '__main__':
    # define the environment
    env_name = TAXI_EXAMPLE
    agent_for_policy_generator = VALUE_ITERATION
    agent_name = Q_LEARNING
    num_of_episodes = 1000
    num_states_in_partial_policy = 10

    # get the environment
    env = get_env(env_name)

    # the anticipated policy (which is part of our input and defined by the user)
    # automatic_anticipated_policy = anticipated_policy_generator.get_automatic_anticipated_policy(env, env_name,
    #                                                                                              agent_for_policy_generator,
    #                                                                                              num_of_episodes,
    #                                                                                              num_states_in_partial_policy)

    anticipated_policy = {(2, 0, None, None, None, None, None, 2): 0}

    # create agent
    agent = rl_agent.create_agent(env, env_name, agent_name)
    # train the agent in the environment
    train_episode_reward_mean = rl_agent.run(agent, num_of_episodes, method=TRAIN)

    # evaluate the performance of the agent
    evaluate_episode_reward_mean = rl_agent.run(agent, num_of_episodes, method=EVALUATE)

    # create a transformed environment
    transforms = [taxi_infinite_fuel_transform]
    explanation = None

    transform_rewards = []
    transformed_env = env
    for transform in transforms:
        # create transformed environment
        transformed_env = transform(transformed_env)

        # create agent
        agent = rl_agent.create_agent(transformed_env, env_name, agent_name)
        # evaluate the performance of the agent
        transform_episode_reward_mean = rl_agent.run(agent, num_of_episodes, method=TRAIN)

        transform_rewards.append(transform_episode_reward_mean)

        # check if the target policy is achieved in trans_env
        if anticipated_policy_achieved(transformed_env, agent, anticipated_policy):
            explanation = transform
            break

    if explanation is None:
        print("no explanation found - you are too dumb for our system")
    else:
        print("explanation found %s:" % explanation)

    # rl_agent.run_episode(transformed_env, agent, number_of_agents, display=True)
    # visualize rewards
    # results = [episode_reward_mean] + transform_rewards
    names = [WITHOUT_TRANSFORM, "no fuel", "rewards"]
    # plot_result_graph(agent_name, results, names, "episode_reward_mean")
