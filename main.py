from Environments.MultiTaxiEnv.multitaxienv.config import NEW_TAXI_ENVIRONMENT_REWARDS
from Observer.anticipated_policy_generator import OptimalAgent
from Transforms.taxi_transforms import *
from utils import *
from visualize import *
from Agents.rl_agent import *
import Agents.rl_agent as rl_agent
import ray

if __name__ == '__main__':
    # define the environment
    env_name = TAXI
    number_of_agents = 1
    agent_name = IMPALA
    iteration_num = 3
    theta = 48
    discount_factor = 0.9

    # get the environment
    env = get_env(env_name, number_of_agents)

    # get the optimal policy
    # optimal_agent = OptimalAgent(env())
    # policy_dict, policy, V = optimal_agent.value_iteration(theta=theta, discount_factor=discount_factor, display=True)

    # define the agents that are operating in the environment
    ray.init(num_gpus=NUM_GPUS, local_mode=True)

    # create agent and train it in env
    agent, episode_reward_mean = rl_agent.create_agent_and_train(env, env_name, number_of_agents, agent_name,
                                                                 iteration_num, display=False)

    # evaluate the performance of the agent
    # rl_agent.run_episode(env, agent, number_of_agents, display=True)  # TODO Mira: add evaluation function?

    # the target policy (which is part of our input and defined by the user)
    target_policy = {
        (0, 0, None, 0, 0, None, None, 2): 4,  # pickup
        (2, 0, None, 2, 0, None, None, 2): 4,  # pickup
        (0, 2, None, 0, 2, None, None, 2): 4,  # pickup
        (2, 2, None, 2, 2, None, None, 2): 4,  # pickup <==
        (0, 0, None, None, None, 0, 0, 3): 5,  # dropoff
        (2, 0, None, None, None, 2, 0, 3): 5,  # dropoff
        (0, 2, None, None, None, 0, 2, 3): 5,  # dropoff
        (2, 2, None, None, None, 2, 2, 3): 5}  # dropoff

    new_reward = dict(
        step=-1,
        no_fuel=-20,
        bad_pickup=-15,
        bad_dropoff=-15,
        bad_refuel=-10,
        bad_fuel=-50,
        pickup=50,
        standby_engine_off=-1,
        turn_engine_on=-10e6,
        turn_engine_off=-10e6,
        standby_engine_on=-1,
        intermediate_dropoff=50,
        final_dropoff=100,
        hit_wall=-2,
        collision=-35,
        collided=-20,
        unrelated_action=-15
    )
    # compare policy with target policy
    # get the policy of the agents for all the states defined in the target policy e.g. [3,3,0,2,3,4,5] [3,3,0,2,3,4,8]
    # TODO Mira: I think we don't need the mapping function here, because the data structure will be too big (?)
    # compare the target policy with the agent's policy

    # create a transformed environment
    transforms = [taxi_deterministic_position_transform]
    explanation = None

    transform_rewards = []
    transformed_env = env
    for transform in transforms:
        # create transformed environment
        transformed_env = transform(transformed_env)
        set_reward_dict = getattr(transformed_env, "set_reward_dict", None)
        if callable(set_reward_dict):
            set_temp_reward_dict(new_reward)

        # create and train agents in env
        agent, transform_episode_reward_mean = rl_agent.create_agent_and_train(transformed_env, env_name,
                                                                               number_of_agents, agent_name,
                                                                               iteration_num, display=False)
        transform_rewards.append(transform_episode_reward_mean)
        transformed_env = transformed_env()
        # rl_agent.run_episode(transformed_env, agent, number_of_agents, max_episode_len, display=True)
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
    names = [WITHOUT_TRANSFORM, "no fuel", "rewards"]
    plot_result_graph(agent_name, results, names, "episode_reward_mean")

    # shut_down
    ray.shutdown()
