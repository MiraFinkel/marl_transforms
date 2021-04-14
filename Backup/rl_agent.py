def run_episode(env, agent, number_of_agents, max_episode_len, display=False):
    env = env()
    env.set_display(display)
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done and max_episode_len >= 0:
        if number_of_agents == 1:  # single agent
            agent_name = list(obs.keys())[0]
            action = agent.compute_action(obs[agent_name])
            obs, reward, done, info = env.step({agent_name: action})
            done = done['__all__']
            episode_reward += reward[agent_name]
        else:  # multi-agent
            action = {}
            for agent_id, agent_obs in obs.items():
                policy_id = g_config['multiagent']['policy_mapping_fn'](agent_id)
                action[agent_id] = agent.compute_action(agent_obs, policy_id=policy_id)
            obs, reward, done, info = env.step(action)
            done = done['__all__']
            episode_reward += sum(reward.values())  # sum up reward for all agents
        max_episode_len -= 1
    return episode_reward

# class RLAgent(Agents.agent.Agent):
#
#     # init agents and their observations
#     def __init__(self, decision_maker, observation=None):
#         self.decision_maker = decision_maker
#         self.observation = observation
#
#     def get_agent(self):
#         return self.decision_maker