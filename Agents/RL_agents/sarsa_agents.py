from Agents.abstract_agent import AbstractAgent
import numpy as np
import keras
from keras.activations import relu, linear
# import lunar_lander as lander
from collections import deque
import gym
import random
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

learning_rate = 0.001
epsilon = 1
gamma = .99
batch_size = 64
memory = deque(maxlen=1000000)
min_eps = 0.01


class KerasSarsaAgent(AbstractAgent):
    def __init__(self, env, timesteps_per_episode=10001):
        super().__init__(env, timesteps_per_episode)
        self.model = self._build_compile_model()
        self.num_episodes = 400
        self.evaluating = False

    def run(self) -> {str: float}:
        """
        The agent's training method.
        Returns: a dictionary - {"episode_reward_mean": __, "episode_reward_min": __, "episode_reward_max": __,
        "episode_len_mean": __}
        """
        np.random.seed(0)
        scores = []
        for i in range(self.num_episodes + 1):
            score = 0
            state = self.env.reset()
            finished = False
            if i != 0 and i % 50 == 0:
                self.model.save(".\saved_models\model_" + str(i) + "_episodes.h5")
            for j in range(3000):
                # state = np.reshape(state, (1, 8))
                if np.random.random() <= epsilon:
                    action = np.random.choice(7)
                else:
                    predict_state = np.reshape(np.array(state), (1, 1))
                    action_values = self.model.predict(predict_state)
                    action = np.argmax(action_values[0])

                if self.evaluating:
                    self.env.render()
                next_state, reward, finished, metadata = self.env.step(action)
                # next_state = np.reshape(next_state, (1, 8))
                memory.append((state, action, next_state, reward, finished))
                self.replay_experiences()
                score += reward
                state = next_state
                if finished:
                    scores.append(score)
                    print("Episode = {}, Score = {}, Avg_Score = {}".format(i, score, np.mean(scores[-100:])))
                    break

    def _build_compile_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_dim=1, activation=relu))
        model.add(keras.layers.Dense(64, activation=relu))
        model.add(keras.layers.Dense(1, activation=linear))
        model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
        return model

    def compute_action(self, state) -> int:
        """
        Computes the best action from a given state.
        Returns: a int that represents the best action.
        """
        pass

    def stop_episode(self):
        pass

    def episode_callback(self, state, action, reward, next_state, terminated):
        pass

    def evaluate(self):
        pass

    def replay_experiences(self):
        if len(memory) >= batch_size:
            sample_choices = np.array(memory)
            mini_batch_index = np.random.choice(len(sample_choices), batch_size)
            # batch = random.sample(memory, batch_size)
            states, actions, next_states, rewards, finishes = [], [], [], [], []
            for index in mini_batch_index:
                states.append(memory[index][0])
                actions.append(memory[index][1])
                next_states.append(memory[index][2])
                rewards.append(memory[index][3])
                finishes.append(memory[index][4])
            states = np.array(states)
            actions = np.array(actions)
            next_states = np.array(next_states)
            rewards = np.array(rewards)
            finishes = np.array(finishes)
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)
            q_vals_next_state = self.model.predict_on_batch(next_states)
            q_vals_target = self.model.predict_on_batch(states)
            max_q_values_next_state = np.amax(q_vals_next_state, axis=1)
            # q_vals_target[np.arange(batch_size), actions] = rewards + gamma * (max_q_values_next_state) * (1 - finishes)
            tmp = rewards + gamma * max_q_values_next_state * (1 - finishes)
            tmp = np.reshape(tmp, q_vals_target[actions].shape)
            q_vals_target[actions] = tmp
            self.model.fit(states, q_vals_target, verbose=0)
            global epsilon
            if epsilon > min_eps:
                epsilon *= 0.996
