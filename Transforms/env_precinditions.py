import copy

import pandas as pd
import numpy as np
from itertools import combinations
# from Environments.SingleTaxiEnv.single_taxi_env import SingleTaxiEnv


def change_representation(state_info_i):
    for k, v in state_info_i.items():
        if len(v[0]) < 4:
            state_info_i[k] = [(0, -1, 0, False)]
    z = pd.DataFrame(state_info_i)
    y = z.to_numpy()[0]
    arr_2d = np.array([*y])
    new_states_vector = arr_2d[:, 1]
    return new_states_vector


def divide_states_for_allowed_and_not_tmp(P, decode, num_actions):
    # EACH INDEX IN THE LIST IS AN ACTION!
    allowed_states = [set() for _ in range(num_actions)]
    not_allowed_states = [set() for _ in range(num_actions)]
    for state, state_info in P.items():
        current_state = state
        current_state_f = decode(current_state)
        z = pd.DataFrame(state_info)
        y = z.to_numpy()[0]
        arr_2d = np.array([*y])  # turn the dictionary to an np array - idx is action and content is the info
        new_states_vector = arr_2d[:, 1]  # take from the numpy array the 1 idx - the next state
        for act, s in enumerate(new_states_vector):
            state_f = np.array(decode(s))
            feature_diff = current_state_f - state_f

            # only to calc zero feture vec
            zero_state = tuple([0] * len(state_f))
            ###
            if tuple(feature_diff) != zero_state:
                allowed_states[act].add(tuple(current_state_f))
            else:
                not_allowed_states[act].add(tuple(current_state_f))

    return allowed_states, not_allowed_states


def divide_states_for_allowed_and_not(P, decode, num_actions):
    # EACH INDEX IN THE LIST IS AN ACTION!
    allowed_states = get_data_structure(num_actions)
    not_allowed_states = get_data_structure(num_actions)
    for state, state_info in P.items():
        for action, action_info in state_info.items():
            action_info_i = refactor_to_dictionary(action_info)
            current_state = state
            current_state_f = decode(current_state)
            new_states_vector = change_representation(action_info_i)
            for act, s in enumerate(new_states_vector):
                if s == -1:
                    continue
                state_f = np.array(decode(s))
                feature_diff = current_state_f - state_f
                # only to calc zero feature vec
                zero_state = tuple([0] * len(state_f))
                if tuple(feature_diff) != zero_state:
                    allowed_states[action][act].add(tuple(current_state_f))
                else:
                    not_allowed_states[action][act].add(tuple(current_state_f))
    return allowed_states, not_allowed_states


def get_data_structure(num_actions):
    tmp = {}
    for act in range(num_actions):
        tmp[act] = [set() for _ in range(num_actions)]
    return tmp


def refactor_to_dictionary(action_info):
    tmp = {}
    for k, v in enumerate(action_info):
        tmp[k] = [v]
    return tmp


def get_allowed_features(action_feature_sets):
    # f_names = ["new_row", "new_col", "new_pass_idx", "dest_idx", "fuel"]
    # a_names = ["SOUTH", "NORTH", "EAST", "WEST", "PICKUP", "DROPOFF", "REFUEL"]
    arr = np.array(list(action_feature_sets[0][0]))
    num_states, num_features = arr.shape
    pred_dict = {}
    for action in range(len(action_feature_sets)):
        pred_dict[action] = {}
        action_info = action_feature_sets[action]
        for prob_action in range(len(action_info)):
            arr = np.array(list(action_info[prob_action]))
            if not arr.any():
                continue
            feature_dict = {}
            for feature_index in range(num_features):
                idx_list = list(combinations([_ for _ in range(num_features)], feature_index + 1))
                for idx in idx_list:
                    idx_array = np.array([arr[:, k] for k in idx]).T
                    f_pred = np.unique(idx_array, axis=0)
                    feature_dict[idx] = f_pred
            pred_dict[action][prob_action] = feature_dict
    return pred_dict


def get_not_allowed_features(action_feature_allowed, action_feature_not_allowed):
    _, num_features = np.array(list(action_feature_allowed[0][0])).shape
    pred_dict = {}
    for action in range(len(action_feature_allowed)):
        pred_dict[action] = {}
        action_info_allowed = action_feature_allowed[action]
        action_info_not_allowed = action_feature_not_allowed[action]
        for prob_action in range(len(action_info_allowed)):
            arr_allowed = np.array(list(action_info_allowed[prob_action]))
            arr_not_allowed = np.array(list(action_info_not_allowed[prob_action]))
            if not arr_allowed.any() or not arr_not_allowed.any():
                continue
            feature_dict = {}
            for feature_index in range(num_features):
                idx_list = list(combinations([_ for _ in range(num_features)], feature_index + 1))
                for idx in idx_list:
                    idx_allowed = np.array([arr_allowed[:, k] for k in idx]).T
                    idx_not_allowed = np.array([arr_not_allowed[:, k] for k in idx]).T
                    f_pred_allowed = np.unique(idx_allowed, axis=0)
                    f_pred_not_allowed = np.unique(idx_not_allowed, axis=0)
                    real_not_allowed = []
                    for not_allowed_pred in f_pred_not_allowed:
                        if not_allowed_pred not in f_pred_allowed:
                            real_not_allowed.append(not_allowed_pred)
                    if len(real_not_allowed) != 0:
                        feature_dict[idx] = np.array(real_not_allowed)
            pred_dict[action][prob_action] = feature_dict
    return pred_dict


def clean_pyramid(features_to_clean):
    features_to_clean = copy.deepcopy(features_to_clean)
    clean_idx = {}
    for action, action_info in features_to_clean.items():
        clean_idx[action] = {}
        for prob_action, prob_action_info in action_info.items():
            clean_idx[action][prob_action] = {}
            for i, (idx_i, idx_val_i) in enumerate(prob_action_info.items()):
                for j, (idx_j, idx_val_j) in enumerate(prob_action_info.items()):
                    if i < j and len(idx_i) != len(idx_j):
                        same_idx = list(filter(lambda x: x in idx_i, list(idx_j)))
                        i_same_idx = [idx_i.index(idx) for idx in same_idx]
                        j_same_idx = [idx_j.index(idx) for idx in same_idx]
                        for val_i in idx_val_i:
                            for k, val_j in enumerate(idx_val_j):
                                is_same_idx = all([val_i[ii] == val_j[jj] for ii in i_same_idx for jj in
                                                   j_same_idx]) if same_idx else False
                                if is_same_idx:
                                    if idx_j not in clean_idx[action]:
                                        clean_idx[action][prob_action][idx_j] = [k]
                                    else:
                                        clean_idx[action][prob_action][idx_j].append(k)
    for action, action_info in clean_idx.items():
        for prob_action, idx_list in action_info.items():
            for (idx_j, k) in idx_list.items():
                features_to_clean[action][prob_action][idx_j] = np.delete(features_to_clean[action][prob_action][idx_j], list(set(k)), axis=0)
                if not features_to_clean[action][prob_action][idx_j].any():
                    del features_to_clean[action][prob_action][idx_j]
    return features_to_clean


class EnvPreconditions:
    def __init__(self, env):
        self.env = env
        # allowed_states, not_allowed_states = divide_states_for_allowed_and_not_tmp(env.P, env.decode, env.num_actions)
        self.allowed_states, self.not_allowed_states = divide_states_for_allowed_and_not(env.P, env.decode, env.num_actions)
        self.allowed_features = get_allowed_features(self.allowed_states)
        not_allowed_features_with_duplicates = get_not_allowed_features(self.allowed_states, self.not_allowed_states)
        self.not_allowed_features = clean_pyramid(not_allowed_features_with_duplicates)

    def is_the_value_valid(self, action, idx, value_to_check):
        idx = (idx,) if isinstance(idx, int) else idx
        value_to_check = [value_to_check] if isinstance(value_to_check, int) else value_to_check
        if idx in self.allowed_features[action].keys():
            if value_to_check in self.allowed_features[action][idx]:
                return True
        return False

    def get_not_allowed_dictionary(self):
        return self.not_allowed_features


if __name__ == '__main__':
    # env_default_values = [0, 0, 0, 1, MAX_FUEL - 1]
    # state_visibility_indexes = []
    # cur_transforms = {STATE_VISIBILITY_TRANSFORM: (state_visibility_indexes, env_default_values),
    #                   ALL_OUTCOME_DETERMINIZATION: False,
    #                   MOST_LIKELY_OUTCOME: True}
    # env = SingleTaxiTransformedEnv(cur_transforms)
    # f_names = ["new_row", "new_col", "new_pass_idx", "dest_idx", "fuel"]
    # a_names = ["SOUTH", "NORTH", "EAST", "WEST", "PICKUP", "DROPOFF", "REFUEL"]

    # taxi_env = SingleTaxiEnv(deterministic=False)
    # preconditions = EnvPreconditions(taxi_env)
    pass
