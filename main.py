from Transforms.taxi_transforms import taxi_infinite_fuel_transform
from utils import *
from visualize import *
from Agents.rl_agent import *
import Agents.rl_agent as rl_agent
import ray

if __name__ == '__main__':
    # define the environment
    env_name = TAXI
    number_of_agents = 1
    agent_name = PPO
    iteration_num = 2
    # the target policy (which is part of our input and defined by the user)
    target_policy = {(0, 1, None, 0, 0, None, None, 2): 3,  # left
                     (1, 0, None, 0, 0, None, None, 2): 0,  # up
                     (0, 0, None, 0, 0, None, None, 2): 4,  # pickup
                     (4, 1, None, 4, 0, None, None, 2): 3,  # left
                     (3, 0, None, 4, 0, None, None, 2): 1,  # down
                     (4, 0, None, 4, 0, None, None, 2): 4,  # pickup
                     (0, 3, None, 0, 4, None, None, 2): 2,  # right
                     (1, 4, None, 0, 4, None, None, 2): 0,  # up
                     (0, 4, None, 0, 4, None, None, 2): 4,  # pickup
                     (4, 2, None, 4, 3, None, None, 2): 2,  # right
                     (3, 3, None, 4, 3, None, None, 2): 1,  # down
                     (4, 4, None, 4, 3, None, None, 2): 3,  # left
                     (4, 3, None, 4, 3, None, None, 2): 4,  # pickup <==
                     (0, 1, None, None, None, 0, 0, 3): 3,  # left
                     (1, 0, None, None, None, 0, 0, 3): 0,  # up
                     (0, 0, None, None, None, 0, 0, 3): 5,  # dropoff
                     (4, 1, None, None, None, 4, 0, 3): 3,  # left
                     (3, 0, None, None, None, 4, 0, 3): 1,  # down
                     (4, 0, None, None, None, 4, 0, 3): 5,  # dropoff
                     (0, 3, None, None, None, 0, 4, 3): 2,  # right
                     (1, 4, None, None, None, 0, 4, 3): 0,  # up
                     (0, 4, None, None, None, 0, 4, 3): 5,  # dropoff
                     (4, 2, None, None, None, 4, 3, 3): 2,  # right
                     (3, 3, None, None, None, 4, 3, 3): 1,  # down
                     (4, 4, None, None, None, 4, 3, 3): 3,  # left
                     (4, 3, None, None, None, 4, 3, 3): 5}  # dropoff

    # get the environment
    env, env_to_agent = get_env(env_name, number_of_agents)

    # define the agents that are operating in the environment
    ray.init(num_gpus=NUM_GPUS, local_mode=True)

    # create agent and train it in env
    agent, episode_reward_mean = rl_agent.create_agent_and_train(env, env_to_agent, env_name, number_of_agents,
                                                                 agent_name, iteration_num, display=False)

    # evaluate the performance of the agent
    rl_agent.run_episode(env, agent, number_of_agents, display=True)  # TODO Mira: add evaluation function?

    # compare policy with target policy
    # get the policy of the agents for all the states defined in the target policy e.g. [3,3,0,2,3,4,5] [3,3,0,2,3,4,8]
    # TODO Mira: I think we don't need the mapping function here, because the data structure will bw too big (?)
    # compare the target policy with the agent's policy

    # create a transformed environment
    transforms = [taxi_infinite_fuel_transform]
    explanation = None

    # transformed_env, transformed_env_to_agent = get_env(env_name, number_of_agents, with_transform=True)
    transform_rewards = []
    transformed_env = env
    for transform in transforms:
        # create transformed environment
        transformed_env = transform(transformed_env)

        # create and train agents in env
        agent, transform_episode_reward_mean = rl_agent.create_agent_and_train(transformed_env, env_to_agent,
                                                                               env_name, number_of_agents, agent_name,
                                                                               iteration_num, display=False)
        transform_rewards.append(transform_episode_reward_mean)
        # check if the target policy is achieved in trans_env
        if target_policy_achieved(transformed_env, agent, target_policy):
            explanation = transform
            break

    if explanation is None:
        print("no explanation found - you are too dumb for our system")
    else:
        print("explanation found %s:" % explanation)

    # rl_agent.run_episode(transformed_env, agent, number_of_agents, display=True)
    # visualize rewards
    results = [episode_reward_mean] + transform_rewards
    names = [WITHOUT_TRANSFORM, "no fuel"]
    plot_result_graph(agent_name, results, names, "episode_reward_mean")

    # shut_down
    ray.shutdown()
