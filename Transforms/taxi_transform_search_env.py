import pickle

from Environments.SingleTaxiEnv.single_taxi_wrapper import *
from save_load_utils import *
from Testing.train_agent import generate_agent, KERAS_DQN
from utils import *

SAVE_PATH = "taxi_example_data/"
TRANSFORMS_PATH = "taxi_example_data/taxi_transformed_env/single_transform_envs/"
ACT = "action"
IDX = "idx"
VAL = "val"


class SingleTaxiTransformSearchEnv(SingleTaxiSimpleEnv):
    def __init__(self):
        super().__init__()
        self.action_translation_dict = {}
        new_act = self.num_actions - 1
        preconditions = load_pkl_file(SAVE_PATH + "taxi_example_preconditions.pkl")
        all_single_transformed_envs = load_pkl_file(SAVE_PATH + "all_single_transformed_envs.pkl")
        for act, act_info in preconditions.not_allowed_features.items():
            for pre_idx, pre_val in act_info.items():
                for val in pre_val:
                    new_act += 1
                    transform_name = f"{act}_{pre_idx}_{val}"
                    if transform_name not in all_single_transformed_envs.keys():
                        continue
                    transformed_env = all_single_transformed_envs[transform_name]
                    self.action_translation_dict[new_act] = transform_name
                    for s, s_info in self.P.items():
                        self.P[s][new_act] = transformed_env.P[s][act]
        self.num_actions = new_act + 1
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distribution)


class SingleTaxiTransformAnticipatedSearchEnv(SingleTaxiSimpleEnv):
    def __init__(self, anticipated_policy):
        super().__init__()
        self.action_translation_dict = {}
        new_act = self.num_actions - 1
        preconditions = load_pkl_file(SAVE_PATH + "taxi_example_preconditions.pkl")
        all_single_transformed_envs = load_pkl_file(SAVE_PATH + "all_single_transformed_envs.pkl")
        decoded_anticipated_states, anticipated_actions = anticipated_policy.keys(), anticipated_policy.values()
        flatten_anticipated_actions = [item for sublist in anticipated_actions for item in sublist]
        for state, state_info in self.P.items():
            new_act = self.num_actions - 1
            for not_allowed_act, not_allowed_act_info in preconditions.not_allowed_features.items():
                for pre_idx, pre_val in not_allowed_act_info.items():
                    for val in pre_val:
                        transform_name = f"{not_allowed_act}_{pre_idx}_{val}"
                        if transform_name not in all_single_transformed_envs.keys():
                            continue
                        new_act += 1
                        transformed_env = all_single_transformed_envs[transform_name]
                        self.action_translation_dict[new_act] = {ACT: not_allowed_act, IDX: pre_idx, VAL: pre_val}
                        self.P[state][new_act] = transformed_env.P[state][not_allowed_act]
            align_with_anticipated, anticipated_act, _ = is_state_align_with_anticipated_policy(self, state,
                                                                                                anticipated_policy)
            if align_with_anticipated:
                next_anticipated_state = self.P[state][anticipated_act][0][1]
                if state != next_anticipated_state:
                    possible_actions = [anticipated_act]
                else:
                    possible_actions = [k for (k, v) in self.action_translation_dict.items() if
                                        v[ACT] == anticipated_act]
                    possible_actions.append(anticipated_act)
                for other_act, other_act_info in self.P[state].items():
                    if other_act not in possible_actions:
                        temp_info = list(other_act_info[0])
                        temp_info[0] = 0.0
                        self.P[state][other_act] = [tuple(temp_info)]
        self.num_actions = new_act + 1
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distribution)

# if __name__ == '__main__':
#     anticipated_policy = {(2, 0, 0, 3, None): [1],
#                           (1, 0, 0, 3, None): [1],
#                           (0, 0, 0, 3, None): [4]}
#     search_env = SingleTaxiTransformAnticipatedSearchEnv(anticipated_policy)
#     file_name = "taxi_example_data/taxi_transformed_env/search_env_without_5.pkl"
#     save_pkl_file(file_name, search_env)
#
#     transform_name = "search_anticipated_env"
#     save_pkl_file(transform_name, search_env)
#     file = open(file_name, "rb")
#     new_env = pickle.load(file)
#     generate_agent(SINGLE_TAXI_EXAMPLE, KERAS_DQN, 100, new_env, transform_name)
#     a = 7
