from Observer.abstract_expert import AbstractExpert
import numpy as np
import json

PARTIAL_POLICY_IDX = 0  # Should be 0 <= i <=5


class LunarLanderExpert(AbstractExpert):
    def __init__(self, env):
        self.env = env

    def get_expert_policy_set(self, state_tuple):
        pass

    def full_expert_policy_dict(self):
        prob_dict_tmp = {}
        prob_dict = {}
        full_expert_policy_dict = {}
        expert_policy_dict = {}

        with open('results/sarsa_data/sarsa_Q_5X4Ys.json') as json_file:
            Q = json.load(json_file)
            for k, v in Q.items():
                state_name = tuple([int(x) for x in (k[1:-3].split(','))])
                if state_name not in prob_dict_tmp.keys():
                    prob_dict_tmp[state_name] = [0] * 4
                prob_dict_tmp[state_name][int(k[-1])] = v
        for k, val in prob_dict_tmp.items():
            max_val = max(val)
            for i in range(len(val)):
                if val[i] == max_val:
                    prob_dict[k] = i
            full_expert_policy_dict[k] = self.heuristic(list(k))
        for key, value in full_expert_policy_dict.items():
            key = list(key)
            state = tuple(key[:PARTIAL_POLICY_IDX] + [None] + key[PARTIAL_POLICY_IDX + 1:])
            if state not in expert_policy_dict.keys():
                expert_policy_dict[state] = value

        return expert_policy_dict

    def heuristic(self, s):
        # Heuristic for:
        # 1. Testing.
        # 2. Demonstration rollout.

        # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
        angle_targ = s[0] * 0.5 + s[2] * 1.0
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        hover_targ = 0.55 * np.abs(s[0])  # target y should be proportional to horizontal offset

        # PID controller: s[4] angle, s[5] angularSpeed
        angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
        # print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

        # PID controller: s[1] vertical coordinate s[3] vertical speed
        hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5
        # print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

        if s[6] or s[7]:  # legs have contact
            angle_todo = 0
            hover_todo = -(s[3]) * 0.5  # override to reduce fall speed, that's all we need after contact

        if self.env.continuous:
            a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
            a = np.clip(a, -1, +1)
        else:
            a = 0
            if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
                a = 2
            elif angle_todo < -0.05:
                a = 3
            elif angle_todo > +0.05:
                a = 1
        return a

    def demo_heuristic_lander(self, env, seed=None, render=False):
        env.seed(seed)
        total_reward = 0
        steps = 0
        s = env.reset()
        while True:
            a = self.heuristic(s)
            s, r, done, info = env.step(a)
            total_reward += r

            if render:
                still_open = env.render()
                if not still_open:
                    break

            if (steps % 20 == 0 or done) and render:
                print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            if done:
                break
        return total_reward
