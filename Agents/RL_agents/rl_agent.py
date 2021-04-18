from Agents.RL_agents.rllib_agents import *
from Agents.RL_agents.q_learning_agents import *

from Agents.value_iteration_agent import ValueIterationAgent

FORMAT_STRING = "{:3d} mean reward: {:6.2f}, variance: {:6.2f}, running time: {:6.2f}"
from constants import *


def get_rl_agent(agent_name, env, env_name=None, env_to_agent=None):  # TODO - fix the environment
    if is_rllib_agent(agent_name):
        agent = get_rllib_agent(agent_name, env_name, env, env_to_agent)

    elif agent_name == HANDS_ON_DQN:
        agent = DQNAgent(env=env)
    elif agent_name == Q_LEARNING:
        agent = QLearningAgent(env=env)
    elif agent_name == VALUE_ITERATION:
        agent = ValueIterationAgent(env=env)
    else:
        raise Exception("Not valid agent name")
    return agent


def run(agent, num_of_episodes, method=TRAIN):
    episode_reward_mean, print_rate = [], 100
    if method == EVALUATE:
        print()
        print("================ EVALUATING ====================")
        print()
    accum_reward, accum_max_reward, accum_min_reward, accum_episode_len = 0, 0, 0, 0
    for it in range(num_of_episodes):
        result = agent.run()
        if (it + 1) % print_rate == 0:
            print(FORMAT_STRING.format(it + 1, accum_reward / print_rate,
                                       (accum_max_reward / print_rate) - (accum_min_reward / print_rate),
                                       accum_episode_len / print_rate
                                       ))
            episode_reward_mean.append(accum_reward / print_rate)
            accum_reward, accum_max_reward, accum_min_reward, accum_episode_len = 0, 0, 0, 0
        accum_reward += result["episode_reward_mean"]
        accum_max_reward += result["episode_reward_max"]
        accum_min_reward += result["episode_reward_min"]
        accum_episode_len += result["episode_len_mean"]
    if method == EVALUATE:
        agent.evaluate()
    return episode_reward_mean


def run_episode(env, agent, method=TRAIN):
    state = env.reset()

    # Initialize variables
    result = {"episode_reward_mean": 0.0, "episode_reward_min": np.inf, "episode_reward_max": -np.inf,
              "episode_len_mean": 0, "total_episode_reward": 0.0}
    total_reward = 0.0
    episode_len = 0
    for timestep in range(agent.timesteps_per_episode):
        if method == EVALUATE:
            env.render()

        # Run Action
        action = agent.compute_action(state)

        # Take action
        next_state, reward, done, info = agent.env.step(action)
        episode_len += 1

        total_reward += reward
        result["episode_reward_max"] = reward if reward > result["episode_reward_max"] else \
            result["episode_reward_max"]
        result["episode_reward_min"] = reward if reward < result["episode_reward_min"] else \
            result["episode_reward_min"]

        terminated = done
        if terminated:
            agent.stop_episode()
            break

        state = agent.episode_callback(state, action, reward, next_state, terminated)
    result["total_episode_reward"] = total_reward
    result["episode_reward_mean"] = total_reward / agent.timesteps_per_episode
    result["episode_len_mean"] = episode_len
    return result


def create_agent(env, agent_name, env_name=None):
    agent = get_rl_agent(agent_name, env)
    return agent


# get the action performed by the agents in each observation
def get_policy_action(agent_rep, obs, reshape=False):
    # [taxi location], [current_fuel], [passengers_start_locations], [destinations], [passengers_status]
    if reshape:
        obs = np.reshape(obs, (1, len(obs)))
    action = agent_rep.compute_action(obs)
    return action


def get_policy_action_partial_obs(agent_rep, partial_obs, reshape=False):
    # [taxi location], [current_fuel], [passengers_start_locations], [destinations], [passengers_status]
    if reshape:
        partial_obs = np.reshape(partial_obs, (1, len(partial_obs)))
    action = agent_rep.compute_action(partial_obs)
    return action
