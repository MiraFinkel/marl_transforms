import copy
import math
import sys
import numpy as np
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
from Environments.ApplePicking.apple_picking_constants import *
from Environments.ApplePicking.BiDict import bidict

MAP = [
    "+---------+",
    "|A: :z: :A|",
    "| : : : : |",
    "|z: : : : |",
    "| : : :z: |",
    "|S:z:A:z:A|",
    "+---------+",
]

global idx


def try_step_south_or_east(place, max_place):
    new_place = min(place + 1, max_place)
    return new_place


def try_step_west_or_north(place, max_place):
    new_place = max(place - 1, max_place)
    return new_place


def apples_valid(a1, a2, a3, a4):
    result = not (
            a1 != 0 and a1 == a2 and a1 == a3 and a1 == a4 and a2 != 0 and a2 == a3 and a2 == a4 and a3 != 0 and a3 == a4)
    tamp_apple_arr = [a1, a2, a3, a4]
    max_apple = max([a1, a2, a3, a4])
    for i in range(1, max_apple):
        if i not in tamp_apple_arr:
            result = False
    return result


class ApplePickingEnv(discrete.DiscreteEnv):
    """
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup an apple
    Rewards:
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, deterministic=True):
        self.deterministic = deterministic
        self.desc = np.asarray(MAP, dtype='c')
        w, h = self.desc.shape
        self.last_action = None
        self.apple_locations, self.thorny_locations, self.start_position = self.get_info_from_map()
        self.original_apple_locations = copy.deepcopy(self.apple_locations)
        self.num_of_picked_apples = 0
        self.num_rows = int(w - 2)
        self.num_columns = int((h - 1) / 2)
        # actually it is 209 but for decoding we need 5 ^ 5
        self.num_states = (self.num_rows * self.num_columns) * int(math.pow(5, 5))
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        self.initial_state_distribution = np.zeros(self.num_states)
        self.num_actions = len(ACTIONS)
        self.translation_dict = bidict()
        self.P = self.build_transition_matrix(deterministic=deterministic)
        # self.initial_state_distribution /= self.initial_state_distribution.sum()
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distribution)

    def build_transition_matrix(self, deterministic=True):
        """
        Build the transition matrix.
        You can work with deterministic environment or stochastic one by changing the flag in the arguments.
        return: dictionary with the transition matrix
        """
        P = {state: {action: [] for action in range(self.num_actions)} for state in range(self.num_states)}
        global idx
        idx = -1
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                for apple1 in range(len(self.apple_locations) + 1):
                    for apple2 in range(len(self.apple_locations) + 1):
                        for apple3 in range(len(self.apple_locations) + 1):
                            for apple4 in range(len(self.apple_locations) + 1):
                                if not apples_valid(apple1, apple2, apple3, apple4):
                                    continue
                                state = self.encode(row, col, apple1, apple2, apple3, apple4)
                                apple_arr = [apple1, apple2, apple3, apple4]
                                num_of_picked_apples = sum(apple_arr)
                                if row == self.start_position[0] and col == self.start_position[
                                    1] and num_of_picked_apples == 0:
                                    self.initial_state_distribution[state] += 1.0
                                for action in range(self.num_actions):
                                    new_row, new_col = row, col
                                    new_apple1, new_apple2, new_apple3, new_apple4 = apple1, apple2, apple3, apple4
                                    new_apple_arr = [new_apple1, new_apple2, new_apple3, new_apple4]
                                    reward = REWARD_DICT[STEP_REWARD]  # default reward when there is no pickup
                                    done = False
                                    collector_loc = (row, col)

                                    if action in [SOUTH, NORTH, WEST, EAST]:
                                        new_row, new_col, reward = self.try_to_move(action, row, col)

                                    elif action == PICKUP:
                                        new_apple_arr, reward, done = self.try_picking_up(collector_loc, done,
                                                                                          new_apple_arr, reward)
                                    self.num_of_picked_apples = num_of_picked_apples
                                    new_state = self.encode(new_row, new_col, *new_apple_arr)
                                    if deterministic:
                                        P[state][action].append((DETERMINISTIC_PROB, new_state, reward, done))
                                    else:
                                        probs = self.get_stochastic_probs(action, row, col, new_state, reward, done)
                                        P[state][action] = probs
        return P

    def encode(self, collector_row, collector_col, apple1, apple2, apple3, apple4):
        # (5), 5, 5, 5, 5, 5
        # (num_rows), num_columns, (len(self.apple_locations) + 1) X 4
        if self.check_if_state_is_legal((collector_row, collector_col, apple1, apple2, apple3, apple4)):
            global idx
            if (collector_row, collector_col, apple1, apple2, apple3, apple4) not in self.translation_dict.values():
                idx += 1
                self.translation_dict[idx] = (collector_row, collector_col, apple1, apple2, apple3, apple4)
            return self.translation_dict[(collector_row, collector_col, apple1, apple2, apple3, apple4)]
        else:
            raise Exception(f"Not legal state{collector_row, collector_col, apple1, apple2, apple3, apple4}")
        # i = collector_row
        # i *= self.num_columns
        # i += collector_col
        # i *= (len(self.apple_locations) + 1)
        # i += apple1
        # i *= (len(self.apple_locations) + 1)
        # i += apple2
        # i *= (len(self.apple_locations) + 1)
        # i += apple3
        # i *= (len(self.apple_locations) + 1)
        # i += apple4
        # return i

    def decode(self, i):
        # (5, 5, 5, 5), 5, (5)
        # (len(self.apple_locations) + 1) X 4, num_columns, (num_rows)
        return self.translation_dict[i]
        # out = []
        # out.append(i % len(self.apple_locations))
        # i = i // len(self.apple_locations)
        # out.append(i % self.num_columns)
        # i = i // self.num_columns
        # out, i = self.decode_helper_append_apple(out, i)
        # out, i = self.decode_helper_append_apple(out, i)
        # out, i = self.decode_helper_append_apple(out, i)
        # out, i = self.decode_helper_append_apple(out, i)
        # out.append(i)
        # assert 0 <= i < self.num_rows
        # # for j in range(len(self.thorny_locations) - 1, 0, -1):
        # #     out.append(self.thorny_locations[j][1])
        # #     out.append(self.thorny_locations[j][0])
        # return list(reversed(out))

    def decode_helper_append_apple(self, out, i):
        out.append(i % (len(self.apple_locations) + 1))
        i = i // (len(self.apple_locations) + 1)
        return out, i

    def check_if_state_is_legal(self, state, return_idxes=False):
        if isinstance(state, int):
            collector_row, collector_col, apple_locations = self.decode(state)  # TODO - bug!!!
        else:
            collector_row, collector_col, apple_locations = state
        not_valid_idx = []
        if collector_row < 0 or collector_row >= self.num_rows:
            not_valid_idx.append(0)
        if collector_col < 0 or collector_col >= self.num_columns:
            not_valid_idx.append(1)
        if apple_locations < 0 or apple_locations > len(self.apple_locations):
            not_valid_idx.append(2)
        state_is_legal = (len(not_valid_idx) == 0)
        if return_idxes:
            return state_is_legal, not_valid_idx
        return state_is_legal

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        collector_row, collector_col, apple_locations = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        def colorize_loc(j, cur_color):
            x, y = self.apple_locations[j]
            if x == -1 and y == -1:
                xc, yc = self.original_apple_locations[j]
                out[1 + xc][2 * yc + 1] = utils.colorize(out[1 + xc][2 * yc + 1], cur_color)

        out[1 + collector_row][2 * collector_col + 1] = utils.colorize(out[1 + collector_row][2 * collector_col + 1],
                                                                       'gray', highlight=True)

        for (di, dj) in self.thorny_locations:
            out[1 + di][2 * dj + 1] = utils.colorize(ul(out[1 + di][2 * dj + 1]), 'green', bold=True)

        colors = ['magenta', 'blue', 'cyan', 'crimson', 'yellow', 'red', 'white']
        for j in range(len(self.apple_locations)):
            colorize_loc(j, colors[j])

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup"][self.lastaction]))
        else:
            outfile.write("\n")
        # print("current state: ", self.decode(self.s), ", last action: ", self.last_action)
        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def reset(self):
        self.s = discrete.categorical_sample(self.isd, self.np_random)
        self.last_action = None
        return int(self.s)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = discrete.categorical_sample([t[0] if len(t) > 0 else 0 for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.last_action = a
        return int(s), r, d, {"prob": p}

    def try_to_move(self, action, row, col):
        new_row, new_col = row, col
        if action == SOUTH:
            new_row = try_step_south_or_east(row, self.max_row)
        elif action == NORTH:
            new_row = try_step_west_or_north(row, 0)
        elif action == EAST and self.no_wall_to_the_right(row, col):
            new_col = try_step_south_or_east(col, self.max_col)
        elif action == WEST and self.no_wall_to_the_left(row, col):
            new_col = try_step_west_or_north(col, 0)
        reward = REWARD_DICT[STEP_INTO_THORN_REWARD] if (new_row, new_col) in self.thorny_locations else REWARD_DICT[
            STEP_REWARD]
        return new_row, new_col, reward

    def get_stochastic_probs(self, action, row, col, pass_idx, dest_idx, fuel, new_state, reward, done):
        if action in [REFUEL, PICKUP, DROPOFF]:
            return self.prob_list_for_no_move_action(action, new_state, reward, done)
        prob_list = [tuple() for _ in range(self.num_actions)]
        action_prob = (STOCHASTIC_PROB, new_state, reward, done)
        prob_list[action] = action_prob
        for prob_act in range(len(prob_list)):
            init_fuel = fuel
            if prob_act != action and prob_act < 4:
                if fuel == 0:
                    new_row, new_col, reward, done = row, col, REWARD_DICT[NO_FUEL_REWARD], True
                else:
                    new_row, new_col, fuel = self.try_to_move(prob_act, fuel, row, col)
                    reward, done = REWARD_DICT[STEP_REWARD], False
                new_state = self.encode(new_row, new_col, pass_idx, dest_idx, fuel)
                prob_list[prob_act] = (STOCHASTIC_PROB_OTHER_ACTIONS, new_state, reward, done)
            fuel = init_fuel
        return prob_list

    def prob_list_for_no_move_action(self, action, new_state, reward, done):
        prob_list = [tuple() for _ in range(self.num_actions)]
        prob_list[action] = (DETERMINISTIC_PROB, new_state, reward, done)
        return prob_list

    def no_wall_to_the_right(self, row, col):
        return self.desc[1 + row, 2 * col + 2] == b":"

    def no_wall_to_the_left(self, row, col):
        return self.desc[1 + row, 2 * col] == b":"

    def try_picking_up(self, collector_loc, done, new_apple_arr, reward):
        for i, apple_loc in enumerate(self.apple_locations):
            if collector_loc == apple_loc:
                last_collected = max(new_apple_arr)
                if last_collected == 4:
                    done = True
                else:
                    new_apple_arr[i] = last_collected + 1
                reward = REWARD_DICT[APPLE_PICKUP_REWARD]
                break
        if reward != REWARD_DICT[APPLE_PICKUP_REWARD]:
            reward = REWARD_DICT[BAD_APPLE_PICKUP_REWARD]
        return new_apple_arr, reward, done

    def get_info_from_map(self):
        apple_locations, thorny_wall_locations, start_position = [], [], None
        h, w = self.desc.shape
        h, w = (h - 2), (w - 2)
        for x in range(1, h + 1):
            for y in range(1, w + 1):
                c = self.desc[x][y]
                if c == b'A':
                    apple_locations.append((x - 1, int(y / 2)))
                elif c == b'z':
                    thorny_wall_locations.append((x - 1, int(y / 2)))
                elif c == b'S':
                    start_position = (x - 1, int(y / 2))

        return apple_locations, thorny_wall_locations, start_position


#
if __name__ == '__main__':
    new_env = ApplePickingEnv(deterministic=True)
    new_env.reset()
    actions = [1, 1, 1, 1, 4, 2, 2, 2, 2, 4, 0, 0, 0, 0, 4, 3, 3, 4]
    all_reward = 0
    for act in actions:
        new_env.render()
        next_s, r, d, prob = new_env.step(act)
        all_reward += r
        print("state:", new_env.decode(next_s))
        print("reward:", r, "done:", d, "prob:", prob)
        print("all_reward:", all_reward)
