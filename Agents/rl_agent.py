import Agents.agent
# import ray
import numpy as np

# ===================== Agents ===================== #
from constants import NUM_GPUS

A2C = "a2c"
A3C = "a3c"
BC = "bc"
DQN = "dqn"
APEX_DQN = "apex_dqn"
IMPALA = "impala"
MARWIL = "marwil"
PG = "pg"
PPO = "ppo"
APPO = "appo"
SAC = "sac"
LIN_UCB = "lin_usb"
LIN_TS = "lin_ts"
# ===================== Agents ===================== #
NUM_WORKERS = 1
WITH_DEBUG = True
TAXI = "taxi"
SPEAKER_LISTENER = "simple_speaker_listener"

agents_gamma = {'taxi_1': 0.85, 'taxi_2': 0.95, 'taxi_3': 0.85, 'taxi_4': 0.95}

FORMAT_STRING = "{:3d} mean reward: {:6.2f}, variance: {:6.2f}, running time: {:6.2f}"


def get_rl_agent(agent_name, config, env_to_agent):
    if agent_name == A2C:
        import ray.rllib.agents.a3c as a2c
        agent = a2c.A2CTrainer(config=config, env=env_to_agent)
    elif agent_name == A3C:
        import ray.rllib.agents.a3c as a3c
        agent = a3c.A3CTrainer(config=config, env=env_to_agent)
    elif agent_name == BC:
        import ray.rllib.agents.marwil as bc
        agent = bc.BCTrainer(config=config, env=env_to_agent)
    elif agent_name == DQN:
        import ray.rllib.agents.dqn as dqn
        agent = dqn.DQNTrainer(config=config, env=env_to_agent)
    elif agent_name == APEX_DQN:
        import ray.rllib.agents.dqn as dqn
        agent = dqn.ApexTrainer(config=config, env=env_to_agent)
    elif agent_name == IMPALA:
        import ray.rllib.agents.impala as impala
        agent = impala.ImpalaTrainer(config=config, env=env_to_agent)
    elif agent_name == MARWIL:
        import ray.rllib.agents.marwil as marwil
        agent = marwil.MARWILTrainer(config=config, env=env_to_agent)
    elif agent_name == PG:
        import ray.rllib.agents.pg as pg
        agent = pg.PGTrainer(config=config, env=env_to_agent)
    elif agent_name == PPO:
        import ray.rllib.agents.ppo as ppo
        agent = ppo.PPOTrainer(config=config, env=env_to_agent)
    elif agent_name == APPO:
        import ray.rllib.agents.ppo as ppo
        agent = ppo.APPOTrainer(config=config, env=env_to_agent)
    elif agent_name == SAC:
        import ray.rllib.agents.sac as sac
        agent = sac.SACTrainer(config=config, env=env_to_agent)
    elif agent_name == LIN_UCB:
        import ray.rllib.contrib.bandits.agents.lin_ucb as lin_ucb
        agent = lin_ucb.LinUCBTrainer(config=config, env=env_to_agent)
    elif agent_name == LIN_TS:
        import ray.rllib.contrib.bandits.agents.lin_ts as lin_ts
        agent = lin_ts.LinTSTrainer(config=config, env=env_to_agent)
    else:
        raise Exception("Not valid agent name")
    return agent


class RLAgent(Agents.agent.Agent):

    # init agents and their observations
    def __init__(self, decision_maker, observation=None):
        self.decision_maker = decision_maker
        self.observation = observation

    def get_agent(self):
        return self.decision_maker


g_config = {}


def train(agent, iteration_num):
    episode_reward_mean = []
    # episode is until the system terminates
    # epoch is a set of episodes
    for it in range(iteration_num):
        result = agent.train()
        print(FORMAT_STRING.format(
            it + 1,
            result["episode_reward_mean"],
            result["episode_reward_max"] - result["episode_reward_min"],
            result["episode_len_mean"]
        ))
        episode_reward_mean.append(result["episode_reward_mean"])

    return episode_reward_mean


def run_episode(env, agent_rep, number_of_agents, display=False):
    env.set_display(display)  # TODO Guy: to add "set_display" to particle environment
    print()
    print(" ===================================================== ")
    print(" ================ STARTING EVALUATION ================ ")
    print(" ===================================================== ")
    print()
    global g_config
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        if number_of_agents == 1:  # single agent
            agent_name = list(obs.keys())[0]
            action = agent_rep.compute_action(obs[agent_name])
            obs, reward, done, info = env.step({agent_name: action})
            done = done['__all__']
            episode_reward += reward[agent_name]
        else:  # multi-agent
            action = {}
            for agent_id, agent_obs in obs.items():
                policy_id = g_config['multiagent']['policy_mapping_fn'](agent_id)
                action[agent_id] = agent_rep.compute_action(agent_obs, policy_id=policy_id)
            obs, reward, done, info = env.step(action)
            done = done['__all__']
            episode_reward += sum(reward.values())  # sum up reward for all agents
    print(episode_reward)


def create_agent_and_train(env, env_to_agent, env_name, number_of_agents, agent_name, iteration_num, display=False):
    # env.set_display(display)  # TODO Guy: to add "set_display" to particle environment
    config = get_config(env_name, env, number_of_agents)
    agent = get_rl_agent(agent_name, config, env_to_agent)

    # train the agent in the environment
    episode_reward_mean = train(agent, iteration_num)
    return agent, episode_reward_mean


def get_config(env_name, env, number_of_agents):
    """
    TODO Guy: to expand the function to work with particle environment
    :param env_name
    :param env:
    :param number_of_agents:
    :return:
    """
    config = {}
    if env_name == TAXI:
        if number_of_agents == 1:  # single agent config
            config = {"num_gpus": NUM_GPUS, "num_workers": NUM_WORKERS}
        else:  # multi-agent config
            policies = get_multi_agent_policies(env, number_of_agents)
            config = {'multiagent': {'policies': policies, "policy_mapping_fn": lambda taxi_id: taxi_id},
                      "num_gpus": NUM_GPUS,
                      "num_workers": NUM_WORKERS}
    if env_name == SPEAKER_LISTENER:
        config = {
            "num_gpus": 0,
            "lr_schedule": [[0, 0.007], [20000000, 0.0000000001]],
            "framework": "torch",
            "env_config": {"name": "simple_speaker_listener"},
            "clip_rewards": True,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 20,
            "monitor": True,
        }
    global g_config
    g_config = config
    return config


def get_multi_agent_policies(env, number_of_agents):
    policies = {}
    for i in range(number_of_agents):
        name = 'taxi_' + str(i + 1)
        policies[name] = (None, env.observation_space, env.action_space, {'gamma': agents_gamma[name]})
    return policies


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
        partial_obs = np.reshape(partial_obs, (1, len(partial_obs)))  # TODO Mira - Add reshape for multi-agent
    action = agent_rep.compute_action(partial_obs)
    return action
