import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
# from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

from constants import *
from Agents.abstract_agent import AbstractAgent


class KerasCEMAgent(AbstractAgent):
    def __init__(self, env, timesteps_per_episode=10001):
        super().__init__(env, timesteps_per_episode)
        self.evaluating = False
        self.action_size = env.action_space.n
        self.state_size = env.num_states
        self.model = self._build_compile_model()
        memory = EpisodeParameterMemory(limit=1000, window_length=1)
        self.agent = CEMAgent(model=self.model, nb_actions=self.action_size, memory=memory, batch_size=50,
                              nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)

    def run(self) -> {str: float}:
        """
        The agent's training method.
        Returns: a dictionary - {"episode_reward_mean": __, "episode_reward_min": __, "episode_reward_max": __,
        "episode_len_mean": __}
        """
        self.agent.compile()
        history = self.agent.fit(self.env, nb_steps=ITER_NUM, visualize=False, verbose=1)
        if len(history.history) > 0:
            episode_reward = history.history["episode_reward"]
            nb_episode_steps = history.history["nb_episode_steps"]
        else:
            episode_reward, nb_episode_steps = [0], [0]  # TODO - placeholder
        result = {EPISODE_REWARD_MEAN: np.array(episode_reward),
                  EPISODE_STEP_NUM_MEAN: np.array(nb_episode_steps),
                  EPISODE_REWARD_MIN: np.empty([]),
                  EPISODE_REWARD_MAX: np.empty([]), EPISODE_VARIANCE: np.empty([])}
        return result

    def _build_compile_model(self):
        ## simple net
        # model = Sequential()
        # model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        # model.add(Dense(self.action_size))
        # model.add(Activation('softmax'))
        # return model
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.add(Activation('softmax'))
        return model

    def compute_action(self, state) -> int:
        """
        Computes the best action from a given state.
        Returns: a int that represents the best action.
        """
        state = np.array([[state]])
        return int(np.argmax(self.model.predict(state)))

    def stop_episode(self):
        pass

    def episode_callback(self, state, action, reward, next_state, terminated):
        pass

    def evaluate(self, visualize=False):
        self.agent.test(self.env, nb_episodes=5, visualize=visualize, nb_max_episode_steps=60)

    def replay_experiences(self):
        pass
