import collections
import copy
import pickle

from Environments.SingleTaxiEnv.single_taxi_wrapper import *
from Environments.frozenlake_environment import FrozenLakeEnv
from Transforms.env_precinditions import EnvPreconditions
from scipy import linalg
from collections import ChainMap
import pandas as pd

SLIPPERY = True
DETERMINISTIC = not SLIPPERY
SOUTH = 1



class FrozenLakeTransformedEnv(FrozenLakeEnv):
    def __init__(self, is_slippery=True, wind=SOUTH):
        super().__init__(is_slippery=is_slippery, wind=wind, )

        self.base_num_actions = self.num_actions


        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.isd)


    def finish_determinazation(self, p, num_actions):

        self.P = p
        self.num_actions = num_actions
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if a not in self.P[s]:
                    p[s][a] = [(1.0, s, -1, False)]  #self.P[s][a] = [(1.0, s, -1, False)]
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.isd)

    def get_max_action_number(self):
        p, num_actions = self.get_all_outcome_determinization_matrix()

    def get_single_outcome_determinization_matrix(self, outcome_idx):
        # assert outcome_idx in [x for x in range(self.)]
        mapped_actions = {}
        self.num_states = self.nrow * self.ncol
        ABNORMAL_STATES = [15] # states like the goal state, where the results are different, should be ignored in the action mapping
        # ABNORMAL_STATES should also include hole states but for now I have not included them because in this particular case it doesn't matter
        num_actions = 0
        p = {}

        for (s, s_outcomes) in self.P.items():
            new_outcomes = {}
            i = 0
            for (a, a_outcomes) in s_outcomes.items():     
                for out_idx, probs in enumerate(a_outcomes):
                    if len(a_outcomes) == 1:
                        # print(":( this is final state lol blat", len(a_outcomes))
                        continue
                    else:
                        pass
                        # if a_outcomes + 
                        
                        
                    if out_idx  == outcome_idx:
                        temp = [(1.0,) + probs[1:]]
                    else:
                        temp = [probs]
                    new_outcomes[i] = temp
                    if s not in ABNORMAL_STATES:
                        if i not in mapped_actions:
                            num_actions +=1
                        mapped_actions[i] = a
                    i += 1
                    
            p[s] = new_outcomes


        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if a not in self.P[s]:
                    p[s][a] = [(1.0, s, -1, False)]  #self.P[s][a] = [(1.0, s, -1, False)]

        # self.mapped_actions = mapped_actions
        # self.num_actions = len(mapped_actions)
        num_actions = len(mapped_actions)
        return p, num_actions

    def get_chosen_action_determinization_matrix(self, action_idx_list):
        for action_idx in action_idx_list:
            assert action_idx in [x for x in range(self.num_actions)], "invalid idx"
        self.num_states = self.nrow * self.ncol
        ABNORMAL_STATES = [15] # states like the goal state, where the results are different, should be ignored in the action mapping
        # ABNORMAL_STATES should also include hole states but for now I have not included them because in this particular case it doesn't matter
        num_actions = self.num_actions + len(action_idx_list)
        p = {}
        action_offsets = [i for i in range(len(action_idx_list))]

        for (s, s_outcomes) in self.P.items():
            # new_outcomes = {}
            i = 0
            outcome_dict = {}
            for (a, a_outcomes) in s_outcomes.items():
                if s not in p:
                        p[s] = {}
                if len(a_outcomes) == 1 or i not in action_idx_list:
                    p[s][a] = a_outcomes
                    continue
                elif i in action_idx_list:

                    for pr,next_s,r,d in a_outcomes:
                        if next_s in outcome_dict:
                            outcome_dict[next_s] += pr
                        else:
                            outcome_dict[next_s] = pr

                    max_s = max(outcome_dict, key=outcome_dict.get)
                    a_outcomes = np.array(a_outcomes)
                    pr,new_s,r,d = a_outcomes[np.where(a_outcomes[:, 1] == max_s)][0]
                    # origiral outcomes
                    p[s][self.num_actions+action_offsets[i]] = a_outcomes

                p[s][int(a)] = [(1,0, max_s,r,d)]
                i+=1


        return p, num_actions
   

    def get_all_outcome_determinization_matrix(self):
        mapped_actions = {}
        self.num_states = self.nrow * self.ncol
        ABNORMAL_STATES = [15] # states like the goal state, where the results are different, should be ignored in the action mapping
        # ABNORMAL_STATES should also include hole states but for now I have not included them because in this particular case it doesn't matter
        num_actions = 0
        p = {}

        for (s, s_outcomes) in self.P.items():
            new_outcomes = {}
            i = 0
            for (a, a_outcomes) in s_outcomes.items():            
                for probs in a_outcomes:
                    temp = [(1.0,) + probs[1:]]
                    new_outcomes[i] = temp
                    if s not in ABNORMAL_STATES:
                        if i not in mapped_actions:
                            num_actions +=1
                        mapped_actions[i] = a
                    i += 1

            p[s] = new_outcomes


        
        # for s in range(self.num_states):
        #     for a in range(self.num_actions):
        #         if a not in self.P[s]:
        #             p[s][a] = [(1.0, s, -1, False)]  #self.P[s][a] = [(1.0, s, -1, False)]

        # self.mapped_actions = mapped_actions
        # self.num_actions = len(mapped_actions)
        num_actions = len(mapped_actions)
        return p, num_actions

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
    a_file.close()
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

def calc_max_actions(self, p):
    self.get_all_outcome_determinization_matrix()

# def get_all_outcome_determinization_matrix(self):
#     mapped_actions = {}
#     self.num_states = self.nrow * self.ncol
#     ABNORMAL_STATES = [15] # states like the goal state, where the results are different, should be ignored in the action mapping
#     # ABNORMAL_STATES should also include hole states but for now I have not included them because in this particular case it doesn't matter
#     num_actions = 0
#     p = {}

#     for (s, s_outcomes) in self.P.items():
#         new_outcomes = {}
#         i = 0
#         for (a, a_outcomes) in s_outcomes.items():            
#             for probs in a_outcomes:
#                 temp = [(1.0,) + probs[1:]]
#                 new_outcomes[i] = temp
#                 if s not in ABNORMAL_STATES:
#                     if i not in mapped_actions:
#                         num_actions +=1
#                     mapped_actions[i] = a
#                 i += 1
#             # print(p[s])
#             # print("---------------------")
#             # print(new_outcomes)
#             # break
#             p[s] = new_outcomes


#     self.num_actions = num_actions
#     for s in range(self.num_states):
#         for a in range(self.num_actions):
#             if a not in self.P[s]:
#                 p[s][a] = [(1.0, s, -1, False)]  #self.P[s][a] = [(1.0, s, -1, False)]


#     self.mapped_actions = mapped_actions
#     self.num_actions = len(mapped_actions)
#     return p
    

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
