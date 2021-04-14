from Agents.RL_agents.rllib_agents import *
from Agents.RL_agents.q_learning_agents import *
import numpy as np

FORMAT_STRING = "{:3d} mean reward: {:6.2f}, variance: {:6.2f}, running time: {:6.2f}"
from constants import *


def get_rl_agent(agent_name, env_name, env, env_to_agent):
    config = get_config(env_name, env, 1) if is_rllib_agent(agent_name) else {}
    if agent_name == RLLIB_A2C:
        import ray.rllib.agents.a3c as a2c
        agent = a2c.A2CTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_A3C:
        import ray.rllib.agents.a3c as a3c
        agent = a3c.A3CTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_BC:
        import ray.rllib.agents.marwil as bc
        agent = bc.BCTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_DQN:
        import ray.rllib.agents.dqn as dqn
        agent = dqn.DQNTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_APEX_DQN:
        import ray.rllib.agents.dqn as dqn
        agent = dqn.ApexTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_IMPALA:
        import ray.rllib.agents.impala as impala
        agent = impala.ImpalaTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_MARWIL:
        import ray.rllib.agents.marwil as marwil
        agent = marwil.MARWILTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_PG:
        import ray.rllib.agents.pg as pg
        agent = pg.PGTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_PPO:
        import ray.rllib.agents.ppo as ppo
        agent = ppo.PPOTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_APPO:
        import ray.rllib.agents.ppo as ppo
        agent = ppo.APPOTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_SAC:
        import ray.rllib.agents.sac as sac
        agent = sac.SACTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_LIN_UCB:
        import ray.rllib.contrib.bandits.agents.lin_ucb as lin_ucb
        agent = lin_ucb.LinUCBTrainer(config=config, env=env_to_agent)
    elif agent_name == RLLIB_LIN_TS:
        import ray.rllib.contrib.bandits.agents.lin_ts as lin_ts
        agent = lin_ts.LinTSTrainer(config=config, env=env_to_agent)
    elif agent_name == HANDS_ON_DQN:
        agent = DQNAgent(env=env_to_agent())
    elif agent_name == Q_LEARNING:
        agent = QLearningAgent(env=env_to_agent())
    else:
        raise Exception("Not valid agent name")
    return agent


def run(agent, num_of_episodes, method=TRAIN):
    episode_reward_mean = []
    for it in range(num_of_episodes):
        result = agent.run()
        print(FORMAT_STRING.format(it + 1, result["episode_reward_mean"],
                                   result["episode_reward_max"] - result["episode_reward_min"],
                                   result["episode_len_mean"]
                                   ))
        episode_reward_mean.append(result["episode_reward_mean"])
    if method == EVALUATE:
        agent.evaluate()
    return episode_reward_mean


def run_episode(env, agent, method=TRAIN):
    state = env.reset()

    # Initialize variables
    result = {"episode_reward_mean": 0.0, "episode_reward_min": np.inf, "episode_reward_max": -np.inf,
              "episode_len_mean": 0}
    total_reward = 0.0

    bar = progressbar.ProgressBar(maxval=agent.timesteps_per_episode / 10,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    episode_len = 0
    for timestep in range(agent.timesteps_per_episode):
        # Run Action
        action = agent.compute_action(state[TAXI_NAME][0])

        # Take action
        next_state, reward, dones, info = agent.env.step({TAXI_NAME: action})
        episode_len += 1

        total_reward += reward[TAXI_NAME]
        result["episode_reward_max"] = reward[TAXI_NAME] if reward[TAXI_NAME] > result["episode_reward_max"] else \
            result["episode_reward_max"]
        result["episode_reward_min"] = reward[TAXI_NAME] if reward[TAXI_NAME] < result["episode_reward_min"] else \
            result["episode_reward_min"]

        terminated = all(list(dones.values()))
        if terminated:
            agent.stop_episode()
            break

        state = agent.episode_callback(state, action, reward, next_state, terminated)

        if timestep % 10 == 0:
            bar.update(timestep / 10 + 1)

    bar.finish()
    result["episode_reward_mean"] = total_reward / agent.timesteps_per_episode
    result["episode_len_mean"] = episode_len
    return result


def create_agent_and_run(env, env_name, agent_name, iteration_num, method=TRAIN, display=False):
    env().set_display(display)
    agent = get_rl_agent(agent_name, env_name, env(), env)

    # train the agent in the environment
    episode_reward_mean = run(agent, iteration_num, method=method)
    return agent, episode_reward_mean


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
