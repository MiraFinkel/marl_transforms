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

    # anticipated_policy = {
    #     (0, 0, None, 0, 0, None, None, 2): 4,  # pickup
    #     (2, 0, None, 2, 0, None, None, 2): 4,  # pickup
    #     (0, 2, None, 0, 2, None, None, 2): 4,  # pickup
    #     (2, 2, None, 2, 2, None, None, 2): 4,  # pickup <==
    #     (0, 0, None, None, None, 0, 0, 3): 5,  # dropoff
    #     (2, 0, None, None, None, 2, 0, 3): 5,  # dropoff
    #     (0, 2, None, None, None, 0, 2, 3): 5,  # dropoff
    #     (2, 2, None, None, None, 2, 2, 3): 5}  # dropoff

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

temp_reward = dict(
    step=-1,
    no_fuel=-20,
    bad_pickup=-30,
    bad_dropoff=-30,
    bad_refuel=-10,
    bad_fuel=-50,
    pickup=50,
    intermediate_dropoff=-30,
    final_dropoff=1000,
    hit_wall=-2,
    unrelated_action=-15,
)

set_reward_dict = getattr(transformed_env, "set_reward_dict", None)
if callable(set_reward_dict):
    set_temp_reward_dict(new_reward)