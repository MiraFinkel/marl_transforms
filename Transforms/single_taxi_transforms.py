import collections
import copy
import pickle

from Environments.SingleTaxiEnv.single_taxi_wrapper import *
from Transforms.env_precinditions import EnvPreconditions
from Transforms.transform_constants import *
from scipy import linalg
from collections import ChainMap

DETERMINISTIC = False


class SingleTaxiTransformedEnv(SingleTaxiSimpleEnv):
    def __init__(self, transforms):
        super().__init__(DETERMINISTIC)
        self.state_visibility_transforms = transforms[STATE_VISIBILITY_TRANSFORM]
        self.all_outcome_determinization = transforms[ALL_OUTCOME_DETERMINIZATION]
        self.most_likely_outcome = transforms[MOST_LIKELY_OUTCOME]
        self.relax_preconditions = transforms[PRECONDITION_RELAXATION]

        if self.all_outcome_determinization:
            get_all_outcome_determinization_matrix(self.P)
        if self.most_likely_outcome:
            self.P = get_most_likely_outcome(self.P)
        if self.relax_preconditions:
            preconditions_relaxation(self.relax_preconditions, self)

    def step(self, a):
        s, r, d, p = super(SingleTaxiTransformedEnv, self).step(a)
        next_state = self.decode(s)
        env_default_values = self.state_visibility_transforms[1]
        for i in self.state_visibility_transforms[0]:
            next_state[i] = env_default_values[i]

        transformed_next_state = self.encode(*next_state)
        return int(transformed_next_state), r, d, {"prob": p}


class PreconditionInfo:
    def __init__(self):
        self.actions = []
        self.prob_actions = None

    # def process_preconditions_info(self, preconditions_info):
    #     for act in preconditions_info:


def preconditions_relaxation(preconditions_info, env, deterministic=DETERMINISTIC):
    # a_file = open("diff_dict.pkl", "rb")
    # diff_dict = pickle.load(a_file)
    # a_file = open("state_by_diff.pkl", "rb")
    # state_by_diff = pickle.load(a_file)
    a_file = open("taxi_preconditions.pkl", "rb")
    preconditions = pickle.load(a_file)
    # preconditions = EnvPreconditions(env)
    diff_dict, state_by_diff, next_state_by_action = get_diff_for_actions(env.P, env)
    for act, act_info in preconditions_info.items():  # TODO - add case when there multiple diff_dict values
        if deterministic:  # deterministic case
            pre_process_info_and_update_p(env, act, diff_dict, act_info, act, preconditions, None)
        else:
            for act_prob, pre_info in act_info.items():
                pre_process_info_and_update_p(env, act_prob, diff_dict, pre_info, act, preconditions, act_prob)


def pre_process_info_and_update_p(env, act_to_diff, diff_dict, pre_info, act, preconditions, act_prob=None):
    diff_vec = diff_dict[act_to_diff][0]
    state_to_replace = get_state_to_replace(env, pre_info)
    mapping_states_dict = get_mapping_states_dict(env, state_to_replace, diff_vec, preconditions, diff_dict[act_to_diff])
    update_p_matrix_by_relax_preconditions(env, mapping_states_dict, act, act_prob)


def get_state_to_replace(env, preconditions_info):
    state_to_replace = []
    for state in env.P.keys():
        decoded_state = np.array(env.decode(state))
        for k, v in preconditions_info.items():
            state_values = decoded_state[np.array(list(k))]  # TODO-check if I need all possible precond. combinations
            if state_values == np.array(v):
                state_to_replace.append(state)
    return state_to_replace


def get_mapping_states_dict(env, state_to_replace, optional_diff_vec, preconditions):
    diff_vec = np.array(diff_vec)
    mapping_states_dict = dict((s, 0) for s in state_to_replace)
    for state in state_to_replace:
        decoded_state = np.array(env.decode(state))
        not_legal_next_state = decoded_state + diff_vec
        state_is_legal, not_valid_idx = env.check_if_state_is_legal(not_legal_next_state, return_idxes=True)
        if not state_is_legal:
            new_not_legal_next_state = copy.deepcopy(not_legal_next_state)
            new_not_legal_next_state[not_valid_idx] = decoded_state[not_valid_idx]
            if (new_not_legal_next_state - decoded_state).any():
                mapping_states_dict[state] = (env.encode(*new_not_legal_next_state))
            else:
                mapping_states_dict[state] = state
    return mapping_states_dict


def update_p_matrix_by_relax_preconditions(env, mapping_states_dict, pre_action, pre_prob_action=None):
    for state in mapping_states_dict:
        if pre_prob_action is not None:
            cur_info = list(env.P[state][pre_action][pre_prob_action])
            cur_info[1] = mapping_states_dict[state]
            env.P[state][pre_action][pre_prob_action] = cur_info
        else:  # deterministic case
            env.P[state][pre_action] = mapping_states_dict[state]


def get_dicts(env, state, act_probs, act, next_state_dict_by_action, diff_dict_by_action, state_by_diff):
    next_state = act_probs[1]
    next_state_dict_by_action[act].add((state, next_state))
    state_diff = np.array(env.decode(next_state)) - np.array(env.decode(state))
    if state_diff.any():
        diff_dict_by_action[act].append(state_diff)
        state_by_diff[act].add(state)
    return next_state_dict_by_action, diff_dict_by_action, state_by_diff


def get_diff_for_actions(p, env):
    next_state_by_action = [set() for act in range(len(p[0]))]
    diff_by_action = [list() for act in range(len(p[0]))]
    state_by_diff = [set() for act in range(len(p[0]))]
    for state, state_info in p.items():
        for act, act_probs in state_info.items():
            if len(act_probs) == 1:  # deterministic case
                next_state_by_action, diff_by_action, state_by_diff = get_dicts(env, state, act_probs[0], act,
                                                                                next_state_by_action, diff_by_action,
                                                                                state_by_diff)
            else:
                for act_prob, act_prob_info in enumerate(act_probs):
                    if len(act_prob_info) > 0:
                        next_state_by_action, diff_by_action, state_by_diff = get_dicts(env, state,
                                                                                        act_prob_info, act_prob,
                                                                                        next_state_by_action,
                                                                                        diff_by_action, state_by_diff)
    diff_dict = {}
    for i, action_diff in enumerate(diff_by_action):
        tmp_diff = [tuple(diff) for diff in action_diff]
        occurrences = collections.Counter(tuple(tmp_diff))
        diff_dict[i] = list(occurrences.keys())
    return diff_dict, state_by_diff, next_state_by_action


def get_most_likely_outcome(p):
    for (s, s_probs) in p.items():
        for (a, a_probs) in s_probs.items():
            probs_list = [prob[0] for prob in a_probs]
            max_prob = max(probs_list)
            max_prob_idx = probs_list.index(max_prob)
            p[s][a] = [tuple([1.0] + list(p[s][a][max_prob_idx])[1:])]
    return p


def get_all_outcome_determinization_matrix(p):
    new_p = {}
    for (s, s_probs) in p.items():
        new_actions = []
        for (a, a_probs) in s_probs.items():
            pass


def get_single_taxi_transform_name(transforms):
    taxi_x_transform, taxi_y_transform = transforms[0], transforms[1]
    pass_loc_transform, pass_dest_transform = transforms[2], transforms[3]
    fuel_transform = transforms[4]
    all_outcome_determinization = transforms[5]
    most_likely_outcome = transforms[6]

    name = ""
    name += TAXI_LOC_X if taxi_x_transform else ""
    name += TAXI_LOC_Y if taxi_y_transform else ""
    name += PASS_LOC if pass_loc_transform else ""
    name += PASS_DEST if pass_dest_transform else ""
    name += FUEL if fuel_transform else ""
    name += ALL_OUTCOME_DETERMINIZATION if all_outcome_determinization else ""
    name += MOST_LIKELY_OUTCOME if most_likely_outcome else ""
    return name


if __name__ == '__main__':
    env_default_values = [0, 0, 0, 1, MAX_FUEL - 1]
    cur_action, prob_action, idx_list, val_list = 2, 2, tuple([4]), [0]
    precondition_relaxation = {0: {prob_action: {idx_list: val_list}},
                               1: {prob_action: {idx_list: val_list}},
                               2: {prob_action: {idx_list: val_list}},
                               3: {prob_action: {idx_list: val_list}}}
    # precondition_relaxation = {0: {idx_list: val_list}}  # deterministic case
    state_visibility_indexes = []
    cur_transforms = {STATE_VISIBILITY_TRANSFORM: (state_visibility_indexes, env_default_values),
                      ALL_OUTCOME_DETERMINIZATION: False,
                      MOST_LIKELY_OUTCOME: False,
                      PRECONDITION_RELAXATION: precondition_relaxation}
    new_env = SingleTaxiTransformedEnv(cur_transforms)
    new_env.s = new_env.encode(4, 2, 2, 0, 1)
    cur_state = new_env.s

    for _ in range(10):
        # a = np.random.randint(0, 6)
        a = 6
        new_env.render()
        print("cur_state: ", new_env.decode(cur_state), " ,next action: ", a)
        next_s, r, d, prob = new_env.step(a)
        print("prob: ", prob['prob']['prob'])
        cur_state = next_s

    passenger_locations, fuel_station = new_env.get_info_from_map()
    a = 7
