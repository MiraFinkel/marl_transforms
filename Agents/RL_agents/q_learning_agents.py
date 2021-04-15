import random
from collections import deque
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
import progressbar
import time
from Agents.RL_agents.rl_agent import *
import Agents.RL_agents.rl_agent as rl_agent
from Agents.abstract_agent import AbstractAgent
import numpy as np
from constants import *

HANDS_ON_DQN = "hands_on_dqn"
Q_LEARNING = "q_learning"
VALUE_ITERATION = "value_iteration"

MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY_RATE = 0.001

TAXI_NAME = "taxi_1"
TERMINAL_STATE = 'TERMINAL_STATE'


def flip_coin(p):
    r = random.random()
    return r < p


class QLearningAgent(AbstractAgent):
    def __init__(self, env, epsilon=1, alpha=0.7, gamma=1, timesteps_per_episode=1000):
        """
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_actions - number of actions in the current environment
        """
        super().__init__(env, timesteps_per_episode)
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.num_training = 500
        self.evaluating = False
        """ Parameters """
        self.alpha = alpha
        self.gamma = gamma

        self.num_actions = env.action_space.n
        self.epsilon = epsilon
        self.q_values = {}
        self.terminal_states = None
        self.episodeRewards = 0
        self.policy_dict = {}

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
        actions = [i for i in range(self.env.action_space.n)]
        q_value_dict = {action: self.get_q_value(state, action) for action in actions}
        max_action = max(q_value_dict, key=q_value_dict.get)
        return max_action

    def compute_action(self, state):
        """
          Computes the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.
          Should use transform_fn if it exist.
        """
        state = state[0] if len(state) == 1 else state
        actions = [i for i in range(self.env.action_space.n)]
        encoded_state = self.env.encode(state)
        if flip_coin(self.epsilon):
            return random.choice(actions)
        max_action = self.get_policy(encoded_state)
        return max_action

    def episode_callback(self, state, action, reward, next_state, terminated):
        self.update_alpha()
        state = self.env.encode(state[TAXI_NAME][0])
        encoded_next_state = self.env.encode(next_state[TAXI_NAME][0])
        reward = reward[TAXI_NAME]
        self.update_q_values(state, action, reward, encoded_next_state)
        return next_state

    def update_q_values(self, state, action, reward, next_state):
        old_value = self.get_q_value(state, action)
        next_max = np.max(self.get_policy(next_state))

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[(state, action)] = new_value

    def update_alpha(self):
        """
        Updates the exploration rate in the end of each episode.
        """
        self.epsilon = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(
            -EXPLORATION_DECAY_RATE * self.episodesSoFar)  # Exploration rate decay

    def start_episode(self):
        self.episodeRewards = 0.0

    def stop_episode(self):
        """
          Called by environment when episode is done
        """
        if not self.evaluating:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.evaluating:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning
        self.start_episode()

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

    def run(self):
        result = rl_agent.run_episode(self.env, self)
        self.episodeRewards = result["total_episode_reward"]
        return result

    def evaluate(self):
        print("================ DISPLAY ====================")
        self.evaluating = True
        result = rl_agent.run_episode(self.env, self, method=EVALUATE)
        return result


class DQNAgent(AbstractAgent):
    def __init__(self, env, timesteps_per_episode=1000, batch_size=32):
        super().__init__(env, timesteps_per_episode)
        self.batch_size = batch_size
        # Initialize attributes
        self._state_size = env.num_states
        self._action_size = env.action_space.n
        self._optimizer = Adam(learning_rate=0.01)

        self.experience_replay = deque(maxlen=2000)

        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1

        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))

    def _build_compile_model(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def compute_action(self, state, predict=False):
        state = state[0] if len(state) == 1 else state
        if np.random.rand() <= self.epsilon and not predict:
            return self.env.action_space.sample()

        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def episode_callback(self, state, action, reward, next_state, terminated):
        next_state = np.reshape(next_state, [1, 1])
        self.store(state, action, reward, next_state, terminated)

        if len(self.experience_replay) > self.batch_size:
            self.retrain(self.batch_size)

        return next_state

    def stop_episode(self):
        self.align_target_model()

    def run(self):
        result = rl_agent.run_episode(self.env, self)
        return result

    def retrain(self, batch_size):
        taxi_name = 'taxi_1'
        minibatch = random.sample(self.experience_replay, batch_size)

        for state, action, reward, next_state, terminated in minibatch:

            target = self.q_network.predict(state[0][0][taxi_name][0])

            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state[0][0]['taxi_1'][0])
                target[0][action] = reward[taxi_name] + self.gamma * np.amax(t)

            self.q_network.fit(state[0][0][taxi_name][0], target, epochs=1, verbose=0)
