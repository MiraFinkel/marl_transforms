from Environments.LunarLander.lunar_lander import *
from Environments.abstract_wrapper_env import AbstractWrapperEnv
import json
from itertools import product


def state_extractor(s):
    state = (min(2, max(-2, int((s[0]) / 0.05))),
             min(2, max(-1, int((s[1]) / 0.1))),
             min(2, max(-2, int((s[2]) / 0.1))),
             min(2, max(-2, int((s[3]) / 0.1))),
             min(2, max(-2, int((s[4]) / 0.1))),
             min(2, max(-2, int((s[5]) / 0.1))),
             int(s[6]),
             int(s[7]))
    return state


class LunarLenderWrapper(LunarLander, AbstractWrapperEnv):
    def __init__(self):
        super().__init__()
        with open('results/sarsa_data/sarsa_Q_5X4Ys.json') as json_file:
            Q = json.load(json_file)
            self.num_states = len(Q)
        self.transform_num = 6

    def reset(self):
        obs = super(LunarLenderWrapper, self).reset()
        encoded_state = self.encode(obs) if not isinstance(obs, int) else obs
        decoded_state = self.decode(encoded_state)
        return encoded_state

    def step(self, action):
        next_state, reward, done, info = super(LunarLenderWrapper, self).step(action)
        encoded_next_state = self.encode(next_state)
        decoded_state = self.decode(encoded_next_state)
        return encoded_next_state, reward, done, info

    def get_states_from_partial_obs(self, partial_obs):
        partial_obs = list(partial_obs)
        x_coordinate = [partial_obs[0]] if (partial_obs[0] is not None) else list(range(-2, 3))
        y_coordinate = [partial_obs[1]] if (partial_obs[1] is not None) else list(range(-2, 3))
        x_horizontal_velocity = [partial_obs[2]] if (partial_obs[2] is not None) else list(range(-2, 3))
        y_vertical_velocity = [partial_obs[3]] if (partial_obs[3] is not None) else list(range(-2, 3))
        orientation_in_space = [partial_obs[4]] if (partial_obs[4] is not None) else list(range(-2, 3))
        angular_velocity = [partial_obs[5]] if (partial_obs[5] is not None) else list(range(-2, 3))
        left_leg = [partial_obs[6]] if (partial_obs[6] is not None) else list(range(2))
        right_leg = [partial_obs[7]] if (partial_obs[7] is not None) else list(range(2))
        states = list(
            product(x_coordinate, y_coordinate, x_horizontal_velocity, y_vertical_velocity, orientation_in_space,
                    angular_velocity, left_leg, right_leg, repeat=1))
        states = [self.encode(state) for state in states]
        return states

    def encode(self, state):
        state = state_extractor(state)
        s = [(state[i] + 2) for i in range(len(state)) if i < 6] + [state[6]] + [state[7]]
        x_coordinate, y_coordinate = s[0], s[1]
        x_horizontal_velocity, y_vertical_velocity = s[2], s[3]
        orientation_in_space = s[4]
        angular_velocity = s[5]
        left_leg, right_leg = s[6], s[7]

        i = x_coordinate

        i *= 5
        i += y_coordinate

        i *= 5
        i += x_horizontal_velocity

        i *= 5
        i += y_vertical_velocity

        i *= 5
        i += orientation_in_space

        i *= 5
        i += angular_velocity

        i *= 2
        i += left_leg

        i *= 2
        i += right_leg
        return i

    def decode(self, i):
        j = i
        out = []

        right_leg = i % 2
        out.append(right_leg)
        i = i // 2

        left_leg = i % 2
        out.append(left_leg)
        i = i // 2

        angular_velocity = i % 5
        out.append(angular_velocity)
        i = i // 5

        orientation_in_space = i % 5
        out.append(orientation_in_space)
        i = i // 5

        y_vertical_velocity = i % 5
        out.append(y_vertical_velocity)
        i = i // 5

        x_horizontal_velocity = i % 5
        out.append(x_horizontal_velocity)
        i = i // 5

        y_coordinate = i % 5
        out.append(y_coordinate)
        i = i // 5

        x_coordinate = i % 5
        out.append(x_coordinate)

        assert 0 <= i < 5
        state = list(reversed(out))
        state = [(state[i] - 2) for i in range(len(state)) if i < 6] + [state[6]] + [state[7]]
        return state
