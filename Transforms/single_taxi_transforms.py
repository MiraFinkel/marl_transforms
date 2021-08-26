import collections

from Environments.SingleTaxiEnv.single_taxi_wrapper import *
from Transforms.transform_constants import *
from save_load_utils import *
from Transforms.env_precinditions import *

DETERMINISTIC = True


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
            self.preconditions = preconditions_relaxation(self.relax_preconditions, self)
            self.preconditions_num = sum([len(self.preconditions.not_allowed_features[k]) for k in
                                          self.preconditions.not_allowed_features.keys()])

    def step(self, a):
        s, r, d, p = super(SingleTaxiTransformedEnv, self).step(a)
        next_state = self.decode(s)
        env_default_values = self.state_visibility_transforms[1]
        for i in self.state_visibility_transforms[0]:
            next_state[i] = env_default_values[i]

        transformed_next_state = self.encode(*next_state)
        return int(transformed_next_state), r, d, p


def preconditions_relaxation(preconditions_info, env, deterministic=DETERMINISTIC):
    # preconditions = EnvPreconditions(env)
    a_file = open(PRECONDITIONS_PATH, "rb")
    preconditions = pickle.load(a_file)

    diff_dict, state_by_diff, next_state_by_action = get_diff_for_actions(env.P, env)
    # a_file = open("taxi_example_data/diff_dict.pkl", "rb")
    # diff_dict = pickle.load(a_file)

    for act, act_info in preconditions_info.items():
        if deterministic:  # deterministic case
            pre_process_info_and_update_p(env, act, diff_dict, act_info, act, preconditions, None)
        else:
            for act_prob, pre_info in act_info.items():
                pre_process_info_and_update_p(env, act_prob, diff_dict, pre_info, act, preconditions, act_prob)
    return preconditions


def pre_process_info_and_update_p(env, act_to_diff, diff_dict, pre_info, act, preconditions, act_prob=None):
    state_to_replace = get_state_to_replace(env, pre_info, act, preconditions)
    mapping_states_dict = get_mapping_states_dict(env, state_to_replace, diff_dict[act_to_diff])
    update_p_matrix_by_relax_preconditions(env, mapping_states_dict, act, act_prob)


def get_state_to_replace(env, preconditions_info, act, preconditions):
    state_to_replace = []
    for state in env.P.keys():
        decoded_state = np.array(env.decode(state))
        for k, v in preconditions_info.items():
            state_values = decoded_state[np.array([k])]
            sufficient_for_action = is_sufficient_for_action(k, v, preconditions.not_allowed_features, act,
                                                             decoded_state)
            if (state_values == np.array(v)).all() and sufficient_for_action:
                state_to_replace.append(state)
    return state_to_replace


def is_sufficient_for_action(relaxed_idx, relaxed_val, not_allowed_features, act, decoded_state):
    result = True
    for key, values in not_allowed_features[act].items():
        for val in values:
            if np.array([decoded_state[list(key)] == val]).all() and (
                    np.array([relaxed_idx != key]).all() or np.array([relaxed_val != val]).all()):
                result = False
                break
        if not result:
            break
    return result


def is_state_need_a_replacement(act, preconditions, preconditions_info, act_prob):
    if act_prob:
        allowed_features_by_action = preconditions.allowed_features[act][act_prob]
    else:
        allowed_features_by_action = preconditions.allowed_features[act]


def get_mapping_states_dict(env, state_to_replace, optional_diff_vec):
    mapping_states_dict = dict((s, 0) for s in state_to_replace)
    for state in state_to_replace:
        decoded_state = np.array(env.decode(state))
        diff_vec = np.array([0] * len(decoded_state))
        if len(optional_diff_vec) > 3:
            for key, val in optional_diff_vec.items():
                val = list(val)[0]
                tmp_diff_vec = np.array(key)
                if (decoded_state + tmp_diff_vec[0])[val[0]] == val[1]:
                    diff_vec = tmp_diff_vec[0]
                    reward = tmp_diff_vec[1]
                    done = tmp_diff_vec[2]
                    break
                else:
                    reward = tmp_diff_vec[1]
                    done = tmp_diff_vec[2]
        else:
            diff_vec = np.array(optional_diff_vec[0][0])
            reward = optional_diff_vec[1]
            done = optional_diff_vec[2]
        not_legal_next_state = decoded_state + diff_vec
        state_is_legal, not_valid_idx = env.check_if_state_is_legal(not_legal_next_state, return_idxes=True)
        if not state_is_legal:
            new_not_legal_next_state = copy.deepcopy(not_legal_next_state)
            new_not_legal_next_state[not_valid_idx] = decoded_state[not_valid_idx]
            if (new_not_legal_next_state - decoded_state).any():
                mapping_states_dict[state] = (env.encode(*new_not_legal_next_state), reward, done)
            else:
                mapping_states_dict[state] = (state, reward, done)
        else:
            mapping_states_dict[state] = (env.encode(*not_legal_next_state), reward, done)
    return mapping_states_dict


def update_p_matrix_by_relax_preconditions(env, mapping_states_dict, pre_action, pre_prob_action=None):
    for state in mapping_states_dict:
        if pre_prob_action is not None:
            cur_info = list(env.P[state][pre_action][pre_prob_action])
            cur_info[1] = mapping_states_dict[state][0]
            cur_info[2] = mapping_states_dict[state][1]
            cur_info[3] = mapping_states_dict[state][2]
            env.P[state][pre_action][pre_prob_action] = cur_info
        else:  # deterministic case
            cur_info = list(env.P[state][pre_action][0])
            cur_info[1] = mapping_states_dict[state][0]
            cur_info[2] = mapping_states_dict[state][1]
            cur_info[3] = mapping_states_dict[state][2]
            env.P[state][pre_action] = [tuple(cur_info)]


def get_dicts(env, state, act_probs, act, next_state_dict_by_action, diff_dict_by_action, state_by_diff):
    next_state = act_probs[1]  # extract next state from the state prob
    reward = act_probs[2]
    done = act_probs[3]
    state_diff = np.array(env.decode(next_state)) - np.array(env.decode(state))
    if state_diff.any():
        next_state_dict_by_action[act].add((state, next_state))
        diff_dict_by_action[act].append([state_diff, reward, done])
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
        tmp_diff = [tuple(diff[0]) for diff in action_diff]
        occurrences = collections.Counter(tuple(tmp_diff))
        occurrences = occurrences.keys()
        reward = action_diff[0][1]
        done = action_diff[0][2] if (i != 5 and reward != 100) else True  # TODO - TEMPORARY! to change
        if len(occurrences) > 1:
            tmp_occ = list(occurrences)
            diff_dict[i] = dict(((x, reward, done), set()) for x in tmp_occ)
            for s, next_state in next_state_by_action[i]:
                for diff in tmp_occ:
                    decoded_state, decoded_next_state = np.array(env.decode(s)), np.array(env.decode(next_state))
                    state_diff = decoded_next_state - decoded_state
                    if not (state_diff - diff).any():
                        idx = np.nonzero(diff)
                        val = decoded_next_state[idx]
                        diff_dict[i][(diff, reward, done)].add((tuple(idx[0]), tuple(val)))
        else:
            diff_dict[i] = (list(occurrences), reward, done)
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
    # this is the function that converts the matrix!
    mapped_actions = {}
    ABNORMAL_STATES = [15]
    # states like the goal state, where the results are different, should be ignored in the action mapping
    # ABNORMAL_STATES should also include hole states but for now I have not included them because in this
    # particular case it doesn't matter

    for (s, s_outcomes) in p.items():
        new_outcomes = {}
        i = 0
        for (a, a_outcomes) in s_outcomes.items():
            for probs in a_outcomes:
                temp = (1.0,) + probs[1:]
                new_outcomes[i] = temp
                if s not in ABNORMAL_STATES:
                    mapped_actions[i] = a
                i += 1
        p[s] = new_outcomes


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


def generate_triple_of_transforms(env_pre):
    dir_name = "triple_transform_envs"
    make_dir(TRAINED_AGENT_SAVE_PATH + dir_name)

    same_precondition = False
    for act1, pre1 in env_pre.not_allowed_features.items():
        for act2, pre2 in env_pre.not_allowed_features.items():
            for act3, pre3 in env_pre.not_allowed_features.items():
                for pre_idx1 in pre1.keys():
                    for pre_idx2 in pre2.keys():
                        for pre_idx3 in pre3.keys():
                            for pre_val1 in pre1[pre_idx1]:
                                for pre_val2 in pre2[pre_idx2]:
                                    for pre_val3 in pre3[pre_idx3]:
                                        if (act1 == act2 and (np.array(pre_idx1 == pre_idx2)).all() and np.array(
                                                (pre_val1 == pre_val2)).all()) or (act1 > act2) or ((act2 == act3 and (
                                                np.array(pre_idx2 == pre_idx3)).all() and np.array(
                                            (pre_val2 == pre_val3)).all()) or (act2 > act3)) or ((act1 == act3 and (
                                                np.array(pre_idx1 == pre_idx3)).all() and np.array(
                                            (pre_val1 == pre_val3)).all()) or (act1 > act3)):
                                            same_precondition = True
                                        if not same_precondition:

                                            env_file_name = f"{TRAINED_AGENT_SAVE_PATH}{dir_name}/{act1}_{pre_idx1}_{pre_val1}_{act2}_{pre_idx2}_{pre_val2}_{act3}_{pre_idx3}_{pre_val3}"
                                            if not os.path.exists(env_file_name + ".pkl"):
                                                print(f"act1: {act1} , pre_idx1: {pre_idx1} , pre_val1: {pre_val1}")
                                                print(f"act2: {act2} , pre_idx2: {pre_idx2} , pre_val2: {pre_val2}")
                                                print(f"act3: {act3} , pre_idx3: {pre_idx3} , pre_val3: {pre_val3}")
                                                print("\n")
                                                precondition = {act1: {pre_idx1: pre_val1},
                                                                act2: {pre_idx2: pre_val2},
                                                                act3: {pre_idx3: pre_val3}}
                                                generate_transformed_env(precondition, env_file_name, save=True)
                                            else:
                                                print(f"file: {env_file_name} already exists")
                                        same_precondition = False


def generate_double_of_transforms(env_pre):
    dir_name = "double_transform_envs"
    make_dir(TRAINED_AGENT_SAVE_PATH + dir_name)

    same_precondition = False
    for act1, pre1 in env_pre.not_allowed_features.items():
        for act2, pre2 in env_pre.not_allowed_features.items():
            for pre_idx1 in pre1.keys():
                for pre_idx2 in pre2.keys():
                    for pre_val1 in pre1[pre_idx1]:
                        for pre_val2 in pre2[pre_idx2]:
                            if (act1 == act2 and (np.array(pre_idx1 == pre_idx2)).all() and np.array(
                                    (pre_val1 == pre_val2)).all()) or (act1 > act2):
                                same_precondition = True
                            if not same_precondition:
                                print(f"act1: {act1} , pre_idx1: {pre_idx1} , pre_val1: {pre_val1}")
                                print(f"act2: {act2} , pre_idx2: {pre_idx2} , pre_val2: {pre_val2}")

                                env_file_name = f"{TRAINED_AGENT_SAVE_PATH}{dir_name}/{act1}_{pre_idx1}_{pre_val1}_{act2}_{pre_idx2}_{pre_val2}"
                                precondition = {act1: {pre_idx1: pre_val1},
                                                act2: {pre_idx2: pre_val2}}
                                generate_transformed_env(precondition, env_file_name, save=True)
                            same_precondition = False


def generate_single_transforms(env_preconditions):
    dir_name = "single_transform_envs"
    make_dir(TRAINED_AGENT_SAVE_PATH + dir_name)

    for act, preconditions in env_preconditions.not_allowed_features.items():
        for pre_idx in preconditions.keys():
            for pre_val in preconditions[pre_idx]:
                print(
                    f"calculating for act1: {act} , pre_idx1: {pre_idx} , pre_val1: {pre_val}")
                env_file_name = f"{TRAINED_AGENT_SAVE_PATH}{dir_name}/{act}_{pre_idx}_{pre_val}"
                precondition = {act: {pre_idx: pre_val}}
                generate_transformed_env(precondition, env_file_name, save=True)


def generate_transformed_env(precondition, env_file_name='', save=True, env_default_values=None):
    cur_transforms = {STATE_VISIBILITY_TRANSFORM: ([], env_default_values),
                      ALL_OUTCOME_DETERMINIZATION: False,
                      MOST_LIKELY_OUTCOME: False,
                      PRECONDITION_RELAXATION: precondition}
    new_env = SingleTaxiTransformedEnv(cur_transforms)
    if save:
        save_pkl_file(env_file_name + ".pkl", new_env)
    return new_env

# if __name__ == '__main__':
#     #     env_default_values = [0, 0, 0, 1, MAX_FUEL - 1]
#     a_file = open("taxi_example_data/taxi_example_preconditions.pkl", "rb")
#     cur_env_preconditions = pickle.load(a_file)
#
#     act1, act2, act3, act4, act5 = 0, 1, 2, 4, 5
#     pre_idx = (4,)
#     pre_val = [0]
#     dir_name = "taxi_example_data/taxi_transformed_env/"
#     env_file_name = f"{act1}_{pre_idx}_{pre_val}_{act2}_{pre_idx}_{pre_val}_{act3}_{pre_idx}_{pre_val}_{act4}_{pre_idx}_{pre_val}_{act5}_{pre_idx}_{pre_val}.pkl"
#     precondition = {act1: {pre_idx: pre_val},
#                     act2: {pre_idx: pre_val},
#                     act3: {pre_idx: pre_val},
#                     act4: {pre_idx: pre_val},
#                     act5: {pre_idx: pre_val},
#                     }
#     env = generate_transformed_env(precondition, env_file_name=dir_name + env_file_name, save=True)
#
#     print("DONE!")
