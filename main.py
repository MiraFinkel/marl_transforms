from utils import *
from visualize import *
import Environments.MultiTaxiEnv.multitaxienv.taxi_environment as taxi_env
import Agents.rl_agent
import ray

if __name__ == '__main__':
    # TODO Guy: to add "set_display" to particle environment


    # define the environment
    taxi_env.set_display(False)
    env_name = TAXI
    number_of_agents = 1
    env, env_to_agent = get_env(env_name, number_of_agents)


    # define the agents that are operating in the environment
    ray.init(num_gpus=NUM_GPUS, local_mode=WITH_DEBUG)

    #todo Mira: move this to a seperate funtion
    # create agent and train it in env
    agent_name = Agents.rl_agent.PPO
    config = get_config(env_name, env, number_of_agents)
    agent = Agents.rl_agent.get_rl_agent(agent_name, config, env_to_agent)

    # train the agent in the environment
    iteration_num = 2
    episode_reward_mean = Agents.rl_agent.train(agent, iteration_num)

    # evaluate the performance of the agent
    taxi_env.set_display(True)
    Agents.rl_agent.run_episode(env, agent, number_of_agents, config)


    # the target policy (which is part of our input and defined by the user)
    target_policy = {}
    target_policy[3,3,dc,dc,dc,dc] = 'up'
    target_policy[4,4,dc,dc,dc,dc] = 'down'


    # compare policy with target policy
    # get the policy of the agents for all the states defined in the target policy e.g. [3,3,0,2,3,4,5] [3,3,0,2,3,4,8] [3,3,0,2,3,4,7]
    # compare the target policy with the agent's policy


    # create a transformed environment
    transforms = []
    transforms.append(delete_relaxation_transform)
    explanation = None
    for transform in transfors:
        # create trasnformed environment
        trans_env = x_transformed(env)
        # create and train agents in env
        # check if the target policy is achieved in trans_env
        # if it is than
        explanation = transform

    if explaination is None:
        print("no explanation found - you are too dumb for our system")
    else:
        print("explanation found %s:"%explanation)



    results = [episode_reward_mean]
    names = [WITHOUT_TRANSFORM]
    plot_result_graph(agent_name, results, names, "episode_reward_mean")

    # episode_reward_mean0 = train(env_name, agent_name, iteration_num, with_transform=True,
    #                              transform_idx=TAXIS_LOC_IDX)
    # episode_reward_mean1 = train(env_name, agent_name, iteration_num, with_transform=True,
    #                              transform_idx=FUELS_IDX)
    # episode_reward_mean2 = train(env_name, agent_name, iteration_num, with_transform=True,
    #                              transform_idx=PASS_START_LOC_IDX)
    # episode_reward_mean3 = train(env_name, agent_name, iteration_num, with_transform=True,
    #                              transform_idx=PASS_DEST_IDX)
    # episode_reward_mean4 = train(env_name, agent_name, iteration_num, with_transform=True,
    #                              transform_idx=PASS_STATUS_IDX)



    # shut_down
    ray.shutdown()
