import pandas as pd
import numpy as np


def calc_action_feature_sets(P, decode, num_actions):
    # EACH INDEX IN THE LIST IS AN ACTION!
    action_feature_sets = [set() for _ in range(num_actions)]  # * num_actions
    for k, v in P.items():
        current_state = k
        current_state_f = decode(current_state)
        z = pd.DataFrame(v)
        y = z.to_numpy()[0]
        arr_2d = np.array([*y])
        new_states_vector = arr_2d[:, 1]
        for action, s in enumerate(new_states_vector):
            state_f = np.array(decode(s))
            feature_diff = current_state_f - state_f

            # only to calc zero feture vec
            zero_state = tuple([0] * len(state_f))
            ###
            if tuple(feature_diff) != zero_state:
                action_feature_sets[action].add(tuple(current_state_f))

    return action_feature_sets


def get_final_pred_dict_from_sets(action_feature_sets, f_names, a_names):
    # f_names = ["new_row", "new_col", "new_pass_idx", "dest_idx", "fuel"]
    # a_names = ["SOUTH", "NORTH", "EAST", "WEST", "PICKUP", "DROPOFF", "REFUEL"]
    arr = np.array(list(action_feature_sets[0]))
    num_states, num_features = arr.shape
    pred_dict = {}
    for action in range(len(a_names)):
        arr = np.array(list(action_feature_sets[action]))
        feature_dict = {}
        for feature_index in range(num_features):
            f_name = f_names[feature_index]
            f_pred = np.unique(arr[:, feature_index])
            feature_dict[f_name] = f_pred
        pred_dict[a_names[action]] = feature_dict
    return pred_dict
