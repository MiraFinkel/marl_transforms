import Agents.agent
import time
import numpy as np
import random
# ===================== Agents ===================== #
from constants import NUM_GPUS

MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY_RATE = 0.001
TERMINAL_STATE = 'TERMINAL_STATE'

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


def flip_coin(p):
    r = random.random()
    return r < p


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


class QLearningAgent:
    def __init__(self, env, num_actions, theta, epsilon=0.9, discount=0.9, alpha=0.81, gamma=0.96, mapping_fn=None):
        """
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_actions - number of actions in the current environment
        """
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.num_training = 1000
        """ Parameters """
        self.theta = theta
        self.alpha = alpha
        self.gamma = gamma

        self.mapping_fn = mapping_fn
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.discount = discount
        self.q_values = {}
        self.terminal_states = None
        self.episodeRewards = 0

        self.env = env

    def get_q_value(self, state, action):
        """
          Returns Q(state,action) or 0.0 if we never seen a state or (state,action) tuple
        """
        if (state, action) in self.q_values:
            return self.q_values[(state, action)]
        return 0.0

    def get_policy(self, state):
        """
          Computes the best action to take in a state.
        """
        actions = self.get_legal_actions(state)
        if len(actions) == 0:  # there are no possible actions
            return None
        q_value_dict = {action: self.get_q_value(state, action) for action in actions}
        max_action = max(q_value_dict, key=q_value_dict.get)
        return max_action

    def action_callback(self, state):
        """
          Computes the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.
          Should use transform_fn if it exist.
        """
        actions = self.get_legal_actions(state)
        # if self.mapping_fn:
        #     return self.mapping_fn(state, actions)
        if len(actions) == 0:  # there are no possible actions
            return None
        if flip_coin(self.epsilon):
            return random.choice(actions)
        max_action = self.get_policy(state)
        return max_action

    def episode_callback(self):
        self.update_alpha()
        self.stop_episode()
        self.start_episode()

    def update_alpha(self):
        """
        Updates the exploration rate in the end of each episode.
        """
        self.alpha = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(
            -EXPLORATION_DECAY_RATE * self.episodesSoFar)  # Exploration rate decay

    def start_episode(self):
        # self.lastState = None
        # self.lastAction = None
        self.episodeRewards = 0.0

    def stop_episode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.num_training:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.num_training:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning

    def get_legal_actions(self, state):
        if self.num_actions == 6 or self.num_actions == 4:
            return [i for i in range(self.num_actions)]
        if state == TERMINAL_STATE:
            return ()
        elif state in self.terminal_states:
            return ('exit',)
        return 'up', 'left', 'down', 'right'

    def set_terminal_states(self, terminal_states):
        self.terminal_states = terminal_states

    def train(self):
        episode_rewards = [0.0]
        agent_rewards = [[0.0] for _ in range(1)]
        final_ep_rewards = []
        final_ag_ep_rewards = [[] for _ in range(1)]

        episode_step = 1
        train_steps = 0

        print("Starting iterations...")
        t_time = time.time()
        for _ in range(num_episodes):
            ep_results = run_episode(self.env, self, 1, max_episode_len, False)

            t_reward, a_rewards, t_steps = ep_results
            train_steps += t_steps

            episode_rewards[-1] += t_reward
            for (idx, a_reward) in enumerate(a_rewards):
                agent_rewards[idx][-1] += a_reward

            self.episode_callback()

            if len(episode_rewards) % save_rate == 0:
                final_ep_rewards.append(np.mean(episode_rewards[-save_rate:]))
                for i, rew in enumerate(agent_rewards):
                    final_ag_ep_rewards[i].append(np.mean(rew[-save_rate:]))

                print("steps: {}, episodes: {}, mean episode reward:{}, time:{}".format(
                    train_steps, len(episode_rewards), final_ep_rewards[-1], time.time() - t_time
                ))

                t_time = time.time()

            episode_rewards.append(0)
            for (idx, a_reward) in enumerate(a_rewards):
                agent_rewards[idx].append(0)

            episode_step += 1

        print("Finished a total of {} episodes.".format(len(episode_rewards)))


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


def run_episode(env, agent_rep, number_of_agents, max_episode_len, display=False):
    env.set_display(display)  # TODO Guy: to add "set_display" to particle environment
    global g_config
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done and max_episode_len >= 0:
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
        max_episode_len -= 1
    return episode_reward


def evaluate(num_episodes, env, agent, number_of_agents, save_rate=10, display=False):
    print()
    print(" ===================================================== ")
    print(" ================ STARTING EVALUATION ================ ")
    print(" ===================================================== ")
    print()
    episode_rewards = [0.0]
    agent_rewards = [[0.0]]
    final_ep_rewards = []

    episode_step = 1
    episode_len = 0

    print("Starting iterations...")
    for i in range(num_episodes):
        ep_results = run_episode(env, agent, number_of_agents, display=display)

        t_reward, a_rewards, t_steps = ep_results

        episode_len = t_steps
        episode_rewards[-1] += t_reward

        for (idx, a_reward) in enumerate(a_rewards):
            agent_rewards[idx][-1] += a_reward
            agent_rewards[idx].append(0)

        agent.reset()

        if len(episode_rewards) % save_rate == 0:
            final_ep_rewards.append(np.mean(episode_rewards[-save_rate:]))
            print(" episode length: {}, total episodes: {}, mean episode reward:{}".format(
                episode_len, len(episode_rewards), final_ep_rewards[-1]
            ))

        episode_rewards.append(0)
        episode_step += 1


def create_agent_and_train(env, env_name, number_of_agents, agent_name, iteration_num, display=False):
    env_to_agent, env = env, env()
    env.set_display(display)  # TODO Guy: to add "set_display" to particle environment
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
    elif env_name == SPEAKER_LISTENER:
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
