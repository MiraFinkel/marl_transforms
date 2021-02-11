import Agents.agent
import ray
import numpy as np
# ===================== Agents ===================== #
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
    def __init__(self, decision_maker, observation = None ):
        self.decision_maker = decision_maker
        self.observation = observation

    def get_agent(self):
        return self.decision_maker


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


def run_episode(env, agent_rep, number_of_agents, config):
    print()
    print(" ===================================================== ")
    print(" ================ STARTING EVALUATION ================ ")
    print(" ===================================================== ")
    print()

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
                policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
                action[agent_id] = agent_rep.compute_action(agent_obs, policy_id=policy_id)
            obs, reward, done, info = env.step(action)
            done = done['__all__']
            episode_reward += sum(reward.values())  # sum up reward for all agents
    print(episode_reward)

# get the action performed by the agents in each observation
def get_policy_action( agent_rep, number_of_agents, obs, reshape = False):

    # [taxi location], [current_fuel], [passengers_start_locations], [destinations], [passengers_status]
    if reshape:
        obs = np.reshape(obs, (1, len(obs)))
    action = agent_rep.compute_action(obs)
    print(action)
    return action

def get_policy_action_partial_obs( agent_rep, number_of_agents, partial_obs, reshape = False):

    # [taxi location], [current_fuel], [passengers_start_locations], [destinations], [passengers_status]
    if reshape:
        obs = np.reshape(obs, (1, len(obs)))
    action = agent_rep.compute_action(obs)
    print(action)
    return action


